from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yahtzee_mlops.agent import AgentConfig, DQNAgent, config_from_checkpoint
from yahtzee_mlops.constants import CATEGORIES
from yahtzee_mlops.game import YahtzeeGame
from yahtzee_mlops.heuristic import HeuristicPlayer, describe_action
from yahtzee_mlops.registry import LocalModelRegistry
from yahtzee_mlops.scoring import face_counts, score_category
from yahtzee_mlops.telemetry import TelemetryStore


TELEMETRY = TelemetryStore(ROOT / "data" / "gameplay.sqlite3")
REGISTRY = LocalModelRegistry(ROOT / "models")

st.set_page_config(page_title="Yahtzee Learning Bot", page_icon="Y", layout="wide")


@st.cache_resource
def load_agent() -> tuple[DQNAgent, bool]:
    agent = DQNAgent(AgentConfig(epsilon_start=0.0, epsilon_min=0.0))
    latest = REGISTRY.latest_checkpoint()
    latest_metrics = ROOT / "models" / "latest_metrics.json"
    model_is_ready = False
    if latest and latest_metrics.exists():
        metrics = json.loads(latest_metrics.read_text(encoding="utf-8"))
        model_is_ready = metrics.get("mean_score", 0) >= 170
    if latest and model_is_ready:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        config = config_from_checkpoint(checkpoint)
        config.epsilon_start = 0.0
        config.epsilon_min = 0.0
        agent = DQNAgent(config)
        agent.load_checkpoint(checkpoint)
        agent.epsilon = 0.0
        return agent, True
    return agent, False


def start_new_game() -> None:
    st.session_state.human = YahtzeeGame(rng=random.Random())
    st.session_state.ai = YahtzeeGame(rng=random.Random())
    st.session_state.game_id = TELEMETRY.start_game(model_version=str(REGISTRY.latest_checkpoint()))
    st.session_state.ai_log = []


if "human" not in st.session_state:
    start_new_game()

agent, has_trained_model = load_agent()
heuristic_player = HeuristicPlayer()

st.title("Yahtzee Learning Bot")
st.caption("Human play is logged locally to data/gameplay.sqlite3. Run the pipeline to train from fresh activity.")
if not has_trained_model:
    st.info("No trained model above the quality gate found yet, so the AI is using a heuristic baseline.")

left, right = st.columns([1, 1])

with st.sidebar:
    st.metric("Your score", st.session_state.human.total_score)
    st.metric("AI score", st.session_state.ai.total_score)
    st.metric("Telemetry events", TELEMETRY.summary()["events"])
    if st.button("New game", use_container_width=True):
        start_new_game()
        st.rerun()


def run_ai_turn() -> None:
    ai: YahtzeeGame = st.session_state.ai
    st.session_state.ai_log = []
    while not ai.is_game_over():
        valid = ai.legal_actions()
        action = agent.select_action(ai.state_vector(), valid, training=False) if has_trained_model else heuristic_player.select_action(ai)
        _, _, _, info = ai.step(action)
        st.session_state.ai_log.append(describe_action(action, ai.dice, info.get("score_delta")))  # type: ignore[arg-type]
        TELEMETRY.log_event(
            st.session_state.game_id,
            ai.turn,
            ai.roll_count,
            "ai",
            action[0],
            {"action": str(action), "dice": ai.dice, "score": ai.total_score},
        )
        if action[0] == "score":
            break


with left:
    human: YahtzeeGame = st.session_state.human
    st.subheader("Your Turn")
    if human.first_roll_of_turn:
        if st.button("Roll", type="primary", use_container_width=True):
            human.step(("roll", (0, 0, 0, 0, 0, 0)))
            TELEMETRY.log_event(st.session_state.game_id, human.turn, human.roll_count, "human", "roll", {"keep": [0] * 6, "dice": human.dice})
            st.rerun()
    else:
        st.write(f"Dice: {human.dice}")
        current_counts = face_counts(human.dice)
        keep = []
        cols = st.columns(6)
        for face in range(1, 7):
            keep.append(cols[face - 1].number_input(f"Keep {face}s", 0, current_counts[face - 1], 0, key=f"keep_{face}"))

        roll_col, score_col = st.columns(2)
        with roll_col:
            if human.roll_count < 3 and st.button("Roll again", use_container_width=True):
                human.step(("roll", tuple(int(value) for value in keep)))
                TELEMETRY.log_event(st.session_state.game_id, human.turn, human.roll_count, "human", "roll", {"keep": keep, "dice": human.dice})
                st.rerun()
        with score_col:
            st.write("Pick a category below.")

        for category in CATEGORIES:
            if human.scorecard[category] is None:
                potential = score_category(category, human.dice)
                if st.button(f"{category.replace('_', ' ').title()}: {potential}", key=f"score_{category}", use_container_width=True):
                    human.step(("score", category))
                    TELEMETRY.log_event(
                        st.session_state.game_id,
                        human.turn,
                        human.roll_count,
                        "human",
                        "score",
                        {"category": category, "potential": potential, "total_score": human.total_score},
                    )
                    if human.is_game_over():
                        TELEMETRY.finish_game(st.session_state.game_id, human.total_score)
                    else:
                        run_ai_turn()
                    st.rerun()

with right:
    st.subheader("AI Turn")
    if st.session_state.ai_log:
        for entry in st.session_state.ai_log:
            st.write(entry)
    else:
        st.info("Waiting for your move.")

st.divider()
score_cols = st.columns(2)
for title, game, col in (("Your Scorecard", st.session_state.human, score_cols[0]), ("AI Scorecard", st.session_state.ai, score_cols[1])):
    with col:
        st.subheader(title)
        for category in CATEGORIES:
            value = game.scorecard[category]
            st.write(f"{category.replace('_', ' ').title()}: {'---' if value is None else value}")
