import numpy as np
import torch
import torch.nn as nn
import random

# Define categories
CATEGORIES = [
    'ones', 'twos', 'threes', 'fours', 'fives', 'sixes',
    'three_of_kind', 'four_of_kind', 'full_house',
    'small_straight', 'large_straight', 'yahtzee', 'chance'
]

# Model Architecture (same as training)
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


def print_dice(dice):
    print("\n  Current Dice:")
    print("  " + " ".join([f"[{d}]" for d in dice]))
    print("  " + " ".join([f" {i+1} " for i in range(5)]))


def print_scorecard(game):
    print("\n" + "="*50)
    print("SCORECARD")
    print("="*50)
    
    print("\nUPPER SECTION:")
    upper_total = 0
    for cat in CATEGORIES[:6]:
        score = game.scorecard[cat]
        if score is not None:
            upper_total += score
            print(f"  {cat.capitalize():20s}: {score:3d}")
        else:
            potential = game.calculate_score(cat) if not game.first_roll_of_turn else 0
            print(f"  {cat.capitalize():20s}: --- (potential: {potential})")
    
    print(f"\n  Upper Total: {upper_total}/63")
    if upper_total >= 63:
        print(f"  Bonus: +35 ✓")
    
    print("\nLOWER SECTION:")
    for cat in CATEGORIES[6:]:
        score = game.scorecard[cat]
        if score is not None:
            print(f"  {cat.replace('_', ' ').title():20s}: {score:3d}")
        else:
            potential = game.calculate_score(cat) if not game.first_roll_of_turn else 0
            print(f"  {cat.replace('_', ' ').title():20s}: --- (potential: {potential})")
    
    print(f"\n  TOTAL SCORE: {game.total_score}")
    print("="*50)


def human_turn(game):
    print("\n" + "="*50)
    print("YOUR TURN")
    print("="*50)
    
    if game.first_roll_of_turn:
        input("\nPress Enter to roll all dice...")
        game.roll_dice()
        print_dice(game.dice)
    
    while game.roll_count < 3:
        print(f"\nRoll {game.roll_count}/3")
        choice = input("\n[R]oll again, [S]core, or [V]iew scorecard? ").strip().lower()
        
        if choice == 'v':
            print_scorecard(game)
            continue
        elif choice == 's':
            break
        elif choice == 'r':
            keep_input = input("Which dice to keep? (e.g., '1 3 5' or 'none'): ").strip()
            
            keep_mask = [0, 0, 0, 0, 0]
            if keep_input.lower() != 'none':
                try:
                    positions = [int(x) - 1 for x in keep_input.split()]
                    for pos in positions:
                        if 0 <= pos < 5:
                            keep_mask[pos] = 1
                except:
                    print("Invalid input. Rolling all dice.")
            
            game.roll_dice(keep_mask)
            print_dice(game.dice)
        else:
            print("Invalid choice. Please enter R, S, or V.")
    
    print("\nAvailable categories:")
    available = [(i, cat) for i, cat in enumerate(CATEGORIES) if game.scorecard[cat] is None]
    
    for i, cat in available:
        potential = game.calculate_score(cat)
        print(f"  {i+1}. {cat.replace('_', ' ').title():20s} (would score: {potential})")
    
    while True:
        try:
            choice = int(input("\nSelect category number: ")) - 1
            if 0 <= choice < len(CATEGORIES) and game.scorecard[CATEGORIES[choice]] is None:
                score = game.score_category(CATEGORIES[choice])
                print(f"\nYou scored {score} points in {CATEGORIES[choice].replace('_', ' ').title()}!")
                break
            else:
                print("Invalid choice or category already used.")
        except:
            print("Invalid input. Please enter a number.")


def ai_turn(game, ai_player):
    """Handle AI player's turn"""
    print("\n" + "="*50)
    print("AI TURN")
    print("="*50)
    
    if game.first_roll_of_turn:
        print("\nAI rolling all dice...")
        game.roll_dice()
        print_dice(game.dice)
        input("Press Enter to continue...")
    
    while True:
        state = game.get_state()
        valid_actions = game.get_valid_actions()
        action = ai_player.get_action(state, valid_actions)
        
        action_type, action_value = action
        
        if action_type == 'roll':
            kept = [i+1 for i, k in enumerate(action_value) if k == 1]
            if kept:
                print(f"\nAI keeping dice: {kept}")
            else:
                print("\nAI rerolling all dice...")
            
            game.roll_dice(action_value)
            print_dice(game.dice)
            input("Press Enter to continue...")
        else:
            score = game.score_category(action_value)
            print(f"\nAI scored {score} points in {action_value.replace('_', ' ').title()}!")
            input("Press Enter to continue...")
            break


def main():
    print("\n" + "="*50)
    print("YAHTZEE vs AI")
    print("="*50)
    
    default_path = r"NickPranskePublic\Yahtzee\yahtzee_model_final.pth"
    model_input = input(f"\nEnter path to your .pth model file (default: {default_path}): ").strip()
    model_path = model_input if model_input else default_path
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ai_player = AIPlayer(model_path, device)
        print(f"\nAI loaded successfully! (Using {device})")
    except Exception as e:
        print(f"\nError loading model: {e}")
        return
    
    human_game = YahtzeeGame()
    ai_game = YahtzeeGame()
    
    print("\nStarting game...")
    
    for turn in range(13):
        print(f"\n{'='*50}")
        print(f"TURN {turn + 1}/13")
        print(f"{'='*50}")
        
        human_turn(human_game)
        print_scorecard(human_game)
        
        ai_turn(ai_game, ai_player)
        
        print("\n" + "="*50)
        print(f"SCORES AFTER TURN {turn + 1}")
        print("="*50)
        print(f"  Your Score: {human_game.total_score}")
        print(f"  AI Score:   {ai_game.total_score}")
        print("="*50)
    
    if human_game.total_score > ai_game.total_score:
        print("\nYou win!")
    elif human_game.total_score < ai_game.total_score:
        print("\nAI wins!")
    else:
        print("\nIt's a tie!")


if __name__ == "__main__":
    main()