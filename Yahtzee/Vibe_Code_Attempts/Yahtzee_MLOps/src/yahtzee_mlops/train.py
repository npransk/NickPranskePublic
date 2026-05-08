"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from yahtzee_mlops.actions import action_index, valid_action_mask
from yahtzee_mlops.agent import AgentConfig, DQNAgent, config_from_checkpoint
from yahtzee_mlops.game import YahtzeeGame
from yahtzee_mlops.heuristic import HeuristicPlayer, RandomPlayer
from yahtzee_mlops.registry import LocalModelRegistry
from yahtzee_mlops.replay import Experience


@dataclass
class TrainConfig:
    episodes: int = 10_000
    eval_games: int = 200
    train_every_steps: int = 1
    seed: int = 42
    model_dir: str = "models"


def play_episode(
    agent: DQNAgent,
    training: bool,
    seed: int | None = None,
    train_every_steps: int = 1,
) -> tuple[int, float, int, float | None]:
    game = YahtzeeGame(rng=random.Random(seed))
    state = game.state_vector()
    total_reward = 0.0
    losses: list[float] = []
    steps = 0

    while not game.is_game_over():
        valid_actions = game.legal_actions()
        action = agent.select_action(state, valid_actions, training=training)
        next_state, reward, done, _ = game.step(action)
        next_valid = game.legal_actions()

        if training:
            agent.remember(
                Experience(
                    state=state,
                    action_index=action_index(action),
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_valid_mask=np.array(valid_action_mask(next_valid), dtype=bool),
                )
            )
            if steps % train_every_steps == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

        state = next_state
        total_reward += reward
        steps += 1

    return game.total_score, total_reward, steps, float(np.mean(losses)) if losses else None


def evaluate(agent: DQNAgent, games: int, seed: int) -> dict[str, float]:
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    scores = [play_episode(agent, training=False, seed=seed + i)[0] for i in range(games)]
    agent.epsilon = old_epsilon
    return {
        "games": float(games),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "over_200_rate": float(np.mean(np.array(scores) >= 200)),
        "over_250_rate": float(np.mean(np.array(scores) >= 250)),
    }


def train(config: TrainConfig, agent_config: AgentConfig | None = None, resume_latest: bool = True) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    registry = LocalModelRegistry(Path(config.model_dir))
    latest = registry.latest_checkpoint()
    checkpoint = None
    if resume_latest and latest is not None:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)

    resolved_agent_config = agent_config or (
        config_from_checkpoint(checkpoint, fallback_seed=config.seed) if checkpoint else AgentConfig(seed=config.seed)
    )
    agent = DQNAgent(resolved_agent_config)
    if checkpoint is not None:
        agent.load_checkpoint(checkpoint)

    rolling_scores: list[int] = []
    rolling_losses: list[float] = []
    for episode in range(1, config.episodes + 1):
        score, _, _, loss = play_episode(
            agent,
            training=True,
            seed=config.seed + episode,
            train_every_steps=config.train_every_steps,
        )
        rolling_scores.append(score)
        if loss is not None:
            rolling_losses.append(loss)
        if episode % 100 == 0:
            recent = rolling_scores[-100:]
            print(
                f"episode={episode} mean100={np.mean(recent):.1f} "
                f"epsilon={agent.epsilon:.3f} loss={np.mean(rolling_losses[-100:]) if rolling_losses else 0:.4f}"
            )

    metrics = evaluate(agent, games=config.eval_games, seed=config.seed + 1_000_000)
    metrics.update(
        {
            "episodes": config.episodes,
            "epsilon": agent.epsilon,
            "train_config": asdict(config),
            "agent_config": asdict(agent.config),
        }
    )
    registered = registry.save(agent.checkpoint(), metrics)
    metrics["run_id"] = registered.run_id
    metrics["checkpoint_path"] = str(registered.checkpoint_path)
    return metrics


def evaluate_player(player: object, games: int, seed: int) -> dict[str, float]:
    scores: list[int] = []
    steps: list[int] = []
    for index in range(games):
        game = YahtzeeGame(rng=random.Random(seed + index))
        step_count = 0
        while not game.is_game_over():
            action = player.select_action(game)  # type: ignore[attr-defined]
            game.step(action)
            step_count += 1
        scores.append(game.total_score)
        steps.append(step_count)

    score_array = np.array(scores)
    return {
        "games": float(games),
        "mean_score": float(score_array.mean()),
        "median_score": float(np.median(score_array)),
        "std_score": float(score_array.std()),
        "min_score": float(score_array.min()),
        "max_score": float(score_array.max()),
        "over_200_rate": float(np.mean(score_array >= 200)),
        "over_250_rate": float(np.mean(score_array >= 250)),
        "mean_steps": float(np.mean(steps)),
    }


def benchmark_baselines(games: int = 1_000, seed: int = 42) -> dict[str, dict[str, float]]:
    return {
        "random": evaluate_player(RandomPlayer(seed=seed), games=games, seed=seed),
        "heuristic": evaluate_player(HeuristicPlayer(), games=games, seed=seed),
    }


def imitate_heuristic(
    episodes: int,
    epochs: int,
    eval_games: int,
    model_dir: str,
    seed: int,
    agent_config: AgentConfig | None = None,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = DQNAgent(agent_config or AgentConfig(seed=seed, epsilon_start=0.0, epsilon_min=0.0))
    teacher = HeuristicPlayer()
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []

    for episode in range(episodes):
        game = YahtzeeGame(rng=random.Random(seed + episode))
        while not game.is_game_over():
            valid = game.legal_actions()
            action = teacher.select_action(game)
            states.append(game.state_vector())
            masks.append(np.array(valid_action_mask(valid), dtype=bool))
            labels.append(action_index(action))
            game.step(action)

    optimizer = torch.optim.AdamW(agent.q_network.parameters(), lr=agent.config.learning_rate)
    batch_size = agent.config.batch_size
    indices = np.arange(len(labels))
    losses: list[float] = []

    for epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_states = torch.as_tensor(np.stack([states[i] for i in batch_indices]), dtype=torch.float32, device=agent.device)
            batch_masks = torch.as_tensor(np.stack([masks[i] for i in batch_indices]), dtype=torch.bool, device=agent.device)
            batch_labels = torch.as_tensor([labels[i] for i in batch_indices], dtype=torch.long, device=agent.device)
            logits = agent.q_network(batch_states).masked_fill(~batch_masks, -1e9)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.q_network.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch + 1} loss={np.mean(losses[-max(1, len(indices) // batch_size):]):.4f}")

    agent.target_network.load_state_dict(agent.q_network.state_dict())
    agent.epsilon = 0.0
    metrics = evaluate(agent, games=eval_games, seed=seed + 2_000_000)
    metrics.update(
        {
            "training_mode": "heuristic_imitation",
            "teacher_episodes": episodes,
            "training_examples": len(labels),
            "epochs": epochs,
            "agent_config": asdict(agent.config),
            "mean_imitation_loss": float(np.mean(losses)) if losses else None,
        }
    )
    registered = LocalModelRegistry(Path(model_dir)).save(agent.checkpoint(), metrics)
    metrics["run_id"] = registered.run_id
    metrics["checkpoint_path"] = str(registered.checkpoint_path)
    return metrics
