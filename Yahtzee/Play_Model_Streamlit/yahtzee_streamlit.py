import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import random
from pathlib import Path

st.set_page_config(page_title="Yahtzee vs AI", page_icon="🎲", layout="wide")

CATEGORIES = [
    'ones', 'twos', 'threes', 'fours', 'fives', 'sixes',
    'three_of_kind', 'four_of_kind', 'full_house',
    'small_straight', 'large_straight', 'yahtzee', 'chance'
]

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DuelingDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class YahtzeeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice = np.zeros(5, dtype=int)
        self.scorecard = {cat: None for cat in CATEGORIES}
        self.roll_count = 0
        self.turn = 0
        self.total_score = 0
        self.yahtzee_bonus_count = 0
        self.first_roll_of_turn = True
        return self.get_state()
    
    def roll_dice(self, keep_mask=None):
        if keep_mask is None:
            keep_mask = [0, 0, 0, 0, 0]
        
        for i in range(5):
            if not keep_mask[i]:
                self.dice[i] = random.randint(1, 6)
        
        self.roll_count += 1
        self.first_roll_of_turn = False
        return self.dice.copy()
    
    def calculate_score(self, category):
        dice = self.dice
        counts = np.bincount(dice, minlength=7)[1:]
        
        if category == 'ones':
            return np.sum(dice == 1)
        elif category == 'twos':
            return np.sum(dice == 2) * 2
        elif category == 'threes':
            return np.sum(dice == 3) * 3
        elif category == 'fours':
            return np.sum(dice == 4) * 4
        elif category == 'fives':
            return np.sum(dice == 5) * 5
        elif category == 'sixes':
            return np.sum(dice == 6) * 6
        elif category == 'three_of_kind':
            return np.sum(dice) if np.max(counts) >= 3 else 0
        elif category == 'four_of_kind':
            return np.sum(dice) if np.max(counts) >= 4 else 0
        elif category == 'full_house':
            return 25 if sorted(counts[counts > 0]) == [2, 3] else 0
        elif category == 'small_straight':
            dice_set = set(dice)
            straights = [{1,2,3,4}, {2,3,4,5}, {3,4,5,6}]
            return 30 if any(s.issubset(dice_set) for s in straights) else 0
        elif category == 'large_straight':
            return 40 if set(dice) in [{1,2,3,4,5}, {2,3,4,5,6}] else 0
        elif category == 'yahtzee':
            return 50 if np.max(counts) == 5 else 0
        elif category == 'chance':
            return np.sum(dice)
        return 0
    
    def score_category(self, category):
        if self.scorecard[category] is not None:
            return -1
        if self.first_roll_of_turn:
            return -1
        
        score = self.calculate_score(category)
        
        is_yahtzee = np.max(np.bincount(self.dice, minlength=7)[1:]) == 5
        if is_yahtzee and self.scorecard['yahtzee'] is not None and self.scorecard['yahtzee'] > 0:
            score += 100
            self.yahtzee_bonus_count += 1
        
        self.scorecard[category] = score
        self.total_score += score
        
        if self.is_game_over():
            upper_score = sum(self.scorecard[cat] for cat in CATEGORIES[:6] if self.scorecard[cat] is not None)
            if upper_score >= 63:
                self.total_score += 35
        
        self.turn += 1
        self.roll_count = 0
        self.first_roll_of_turn = True
        
        return score
    
    def is_game_over(self):
        return all(score is not None for score in self.scorecard.values())
    
    def get_state(self):
        dice_onehot = np.zeros(30)
        for i, die in enumerate(self.dice):
            if die > 0:
                dice_onehot[i * 6 + die - 1] = 1
        
        scorecard_filled = np.array([1.0 if self.scorecard[cat] is not None else 0.0 for cat in CATEGORIES])
        
        potential_scores = np.zeros(13)
        if not self.first_roll_of_turn:
            for i, cat in enumerate(CATEGORIES):
                if self.scorecard[cat] is None:
                    potential_scores[i] = self.calculate_score(cat) / 50.0
        
        upper_scores = np.array([
            (self.scorecard[cat] if self.scorecard[cat] is not None else 0) / 18.0
            for cat in CATEGORIES[:6]
        ])
        
        roll_count = np.array([self.roll_count / 3.0])
        turn_progress = np.array([self.turn / 13.0])
        upper_sum = sum(self.scorecard[cat] if self.scorecard[cat] is not None else 0
                       for cat in CATEGORIES[:6])
        upper_progress = np.array([min(upper_sum / 63.0, 1.5)])
        has_rolled = np.array([0.0 if self.first_roll_of_turn else 1.0])
        score_normalized = np.array([self.total_score / 400.0])
        yahtzee_bonuses = np.array([self.yahtzee_bonus_count / 3.0])
        
        return np.concatenate([
            dice_onehot, scorecard_filled, potential_scores, upper_scores,
            roll_count, turn_progress, upper_progress, has_rolled,
            score_normalized, yahtzee_bonuses
        ])
    
    def get_valid_actions(self):
        actions = []
        
        if self.first_roll_of_turn:
            actions.append(('roll', tuple([0, 0, 0, 0, 0])))
            return actions
        
        if self.roll_count < 3:
            for i in range(32):
                keep_mask = tuple((i >> j) & 1 for j in range(5))
                actions.append(('roll', keep_mask))
        
        for cat in CATEGORIES:
            if self.scorecard[cat] is None:
                actions.append(('score', cat))
        
        return actions


class AIPlayer:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        state_size = 68
        action_size = 45
        
        self.q_network = DuelingDQN(state_size, action_size, hidden_size=512).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_network.eval()
        
        self.action_map = self._create_action_map()
    
    def _create_action_map(self):
        action_map = []
        for i in range(32):
            keep_mask = tuple((i >> j) & 1 for j in range(5))
            action_map.append(('roll', keep_mask))
        for cat in CATEGORIES:
            action_map.append(('score', cat))
        return action_map
    
    def get_action(self, state, valid_actions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        valid_indices = [self.action_map.index(a) for a in valid_actions]
        masked_q = np.full(len(self.action_map), -1e10)
        masked_q[valid_indices] = q_values[valid_indices]
        
        best_action_idx = np.argmax(masked_q)
        return self.action_map[best_action_idx]

if 'human_game' not in st.session_state:
    st.session_state.human_game = YahtzeeGame()
    st.session_state.ai_game = YahtzeeGame()
    st.session_state.ai_player = None
    st.session_state.game_started = False
    st.session_state.waiting_for_score = False
    st.session_state.ai_log = []

if st.session_state.ai_player is None:
    model_path = Path(__file__).resolve().parent.parent / "yahtzee_model_final.pth"
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.session_state.ai_player = AIPlayer(model_path, device)
    except Exception as e:
        st.error(f"Error loading AI model: {e}")
        st.stop()

st.title("Yahtzee vs AI")
st.markdown("---")

with st.sidebar:
    st.header("Game Info")
    st.metric("Turn", f"{st.session_state.human_game.turn + 1}/13")
    st.metric("Roll", f"{st.session_state.human_game.roll_count}/3")
    
    st.markdown("---")
    st.header("Scores")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Your Score", st.session_state.human_game.total_score)
    with col2:
        st.metric("AI Score", st.session_state.ai_game.total_score)
    
    if st.session_state.human_game.is_game_over():
        st.markdown("---")
        st.header("Game Over!")
        human_score = st.session_state.human_game.total_score
        ai_score = st.session_state.ai_game.total_score
        
        if human_score > ai_score:
            st.success(f"You Win! {human_score} - {ai_score}")
        elif human_score < ai_score:
            st.error(f"AI Wins! {ai_score} - {human_score}")
        else:
            st.info(f"Tie Game! {human_score} - {ai_score}")
        
        if st.button("New Game", use_container_width=True):
            st.session_state.human_game = YahtzeeGame()
            st.session_state.ai_game = YahtzeeGame()
            st.session_state.waiting_for_score = False
            st.session_state.ai_log = []
            st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.header("Your Turn")
    
    if not st.session_state.human_game.first_roll_of_turn:
        st.subheader("Current Dice")
        dice_cols = st.columns(5)
        for i, die in enumerate(st.session_state.human_game.dice):
            dice_cols[i].markdown(f"<h2 style='text-align: center'>{die}</h2>", unsafe_allow_html=True)
    
    if st.session_state.human_game.first_roll_of_turn:
        if st.button("Roll Dice", use_container_width=True, type="primary"):
            st.session_state.human_game.roll_dice()
            st.rerun()

    elif st.session_state.human_game.roll_count < 3 and not st.session_state.waiting_for_score:
        st.subheader("Keep dice?")
        keep_cols = st.columns(5)
        keep_mask = []
        for i in range(5):
            keep = keep_cols[i].checkbox(f"Keep", key=f"keep_{i}")
            keep_mask.append(1 if keep else 0)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Roll Again", use_container_width=True):
                st.session_state.human_game.roll_dice(keep_mask)
                st.rerun()
        with col_b:
            if st.button("Score", use_container_width=True):
                st.session_state.waiting_for_score = True
                st.rerun()
    
    if st.session_state.waiting_for_score or st.session_state.human_game.roll_count >= 3:
        st.subheader("Select Category")
        
        available_cats = [(cat, st.session_state.human_game.calculate_score(cat)) 
                         for cat in CATEGORIES if st.session_state.human_game.scorecard[cat] is None]
        
        for cat, potential in available_cats:
            if st.button(f"{cat.replace('_', ' ').title()}: {potential} pts", 
                        key=f"score_{cat}", use_container_width=True):
                score = st.session_state.human_game.score_category(cat)
                st.session_state.waiting_for_score = False
                
                st.session_state.ai_log = []
                ai_game = st.session_state.ai_game
                ai_player = st.session_state.ai_player
                
                if ai_game.first_roll_of_turn:
                    ai_game.roll_dice()
                    st.session_state.ai_log.append(f"AI rolled: {ai_game.dice}")
                
                while True:
                    state = ai_game.get_state()
                    valid_actions = ai_game.get_valid_actions()
                    action = ai_player.get_action(state, valid_actions)
                    
                    action_type, action_value = action
                    
                    if action_type == 'roll':
                        kept = [i+1 for i, k in enumerate(action_value) if k == 1]
                        if kept:
                            st.session_state.ai_log.append(f"AI kept dice: {kept}")
                        else:
                            st.session_state.ai_log.append("AI rerolled all")
                        
                        ai_game.roll_dice(action_value)
                        st.session_state.ai_log.append(f"Result: {ai_game.dice}")
                    else:
                        ai_score = ai_game.score_category(action_value)
                        st.session_state.ai_log.append(f"AI scored {ai_score} in {action_value.replace('_', ' ').title()}")
                        break
                
                st.rerun()

with col2:
    st.header("AI Turn")
    
    if st.session_state.ai_log:
        for log in st.session_state.ai_log:
            st.write(log)
    else:
        st.info("Waiting for your move...")

st.markdown("---")
st.header("Scorecards")

score_col1, score_col2 = st.columns(2)

with score_col1:
    st.subheader("Your Scorecard")
    
    st.markdown("**Upper Section**")
    upper_total = 0
    for cat in CATEGORIES[:6]:
        score = st.session_state.human_game.scorecard[cat]
        if score is not None:
            upper_total += score
            st.write(f"{cat.capitalize()}: **{score}**")
        else:
            potential = st.session_state.human_game.calculate_score(cat) if not st.session_state.human_game.first_roll_of_turn else 0
            st.write(f"{cat.capitalize()}: --- (potential: {potential})")
    
    st.write(f"**Upper Total: {upper_total}/63**")
    if upper_total >= 63:
        st.success("Bonus: +35")
    
    st.markdown("**Lower Section**")
    for cat in CATEGORIES[6:]:
        score = st.session_state.human_game.scorecard[cat]
        if score is not None:
            st.write(f"{cat.replace('_', ' ').title()}: **{score}**")
        else:
            potential = st.session_state.human_game.calculate_score(cat) if not st.session_state.human_game.first_roll_of_turn else 0
            st.write(f"{cat.replace('_', ' ').title()}: --- (potential: {potential})")

with score_col2:
    st.subheader("AI Scorecard")
    
    st.markdown("**Upper Section**")
    ai_upper_total = 0
    for cat in CATEGORIES[:6]:
        score = st.session_state.ai_game.scorecard[cat]
        if score is not None:
            ai_upper_total += score
            st.write(f"{cat.capitalize()}: **{score}**")
        else:
            st.write(f"{cat.capitalize()}: ---")
    
    st.write(f"**Upper Total: {ai_upper_total}/63**")
    if ai_upper_total >= 63:
        st.success("Bonus: +35")
    
    st.markdown("**Lower Section**")
    for cat in CATEGORIES[6:]:
        score = st.session_state.ai_game.scorecard[cat]
        if score is not None:
            st.write(f"{cat.replace('_', ' ').title()}: **{score}**")
        else:
            st.write(f"{cat.replace('_', ' ').title()}: ---")