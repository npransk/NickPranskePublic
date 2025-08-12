import heapq

# Define adjacency map for the dodecahedron puzzle
adjacency_map = {
    0: [1, 5, 6, 8, 11],
    1: [0, 2, 5, 8, 9],
    2: [1, 5, 7, 9, 10],
    3: [4, 6, 7, 10, 11],
    4: [3, 6, 7, 8, 9],
    5: [0, 1, 2, 10, 11],
    6: [0, 3, 4, 8, 11],
    7: [2, 3, 4, 9, 10],
    8: [0, 1, 4, 6, 9],
    9: [1, 2, 4, 7, 8],
    10: [2, 3, 5, 7, 11],
    11: [0, 3, 5, 6, 10]
}

# List of slot names
slots = ["black", "dark_blue", "light_blue", "light_green", "orange", "pink",
         "purple", "red", "teal", "turquoise", "white", "yellow"]

# List of ball colors (including 'empty')
balls = ["black", "dark_blue", "light_blue", "light_green", "orange", "pink",
         "purple", "red", "teal", "turquoise", "white", "yellow", "empty"]

def getInitialState(slots, balls):
    print("\nSet up your puzzle by selecting which ball is in each slot.")
    print("Available ball colors:")
    for i, ball in enumerate(balls):
        print(f"{i}: {ball}")

    initial_state = []

    for slot in slots:
        while True:
            try:
                choice = int(input(f"\nSelect the ball for the '{slot}' slot (enter number 0-{len(balls)-1}): "))
                if 0 <= choice < len(balls):
                    ball_color = balls[choice]
                    initial_state.append((slot, ball_color))
                    break
                else:
                    print(f"Invalid number. Please enter a number between 0 and {len(balls)-1}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    
    return initial_state

## CHEATER'S VERSION: 1 MOVE WIN ###
initial_state_cheater = [
    ('black', 'black'), 
    ('dark_blue', 'dark_blue'), 
    ('light_blue', 'light_blue'), 
    ('light_green', 'light_green'), 
    ('orange', 'orange'), 
    ('pink', 'pink'), 
    ('purple', 'purple'), 
    ('red', 'red'), 
    ('teal', 'teal'), 
    ('turquoise', 'turquoise'), 
    ('white', 'yellow'), 
    ('yellow', 'empty')
]

## TEST VERSION ###
initial_state_test = [
    ('black', 'pink'), 
    ('dark_blue', 'black'), 
    ('light_blue', 'dark_blue'), 
    ('light_green', 'yellow'), 
    ('orange', 'turquoise'), 
    ('pink', 'purple'), 
    ('purple', 'teal'), 
    ('red', 'light_blue'), 
    ('teal', 'orange'), 
    ('turquoise', 'red'), 
    ('white', 'empty'), 
    ('yellow', 'light_green')
    ]

# Heuristic: count mismatches
def heuristic(state):
    return sum(1 for (slot, ball) in state if slot != "white" and slot != ball)

# Find index of the empty slot
def find_empty(state):
    for i, (_, ball) in enumerate(state):
        if ball == "empty":
            return i
    return -1

# Generate neighbors by swapping empty with adjacent slots
def get_neighbors(state):
    empty_index = find_empty(state)
    neighbors = []
    for adj in adjacency_map[empty_index]:
        new_state = state.copy()
        new_state[empty_index], new_state[adj] = (new_state[empty_index][0], new_state[adj][1]), (new_state[adj][0], "empty")
        neighbors.append((new_state, adj))
    return neighbors

# A* search
def a_star(start, goal):
    frontier = []
    heapq.heappush(frontier, (heuristic(start), 0, start, []))
    visited = set()

    while frontier:
        _, cost, current, path = heapq.heappop(frontier)
        state_tuple = tuple((slot, ball) for slot, ball in current)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        if current == goal:
            return path + [current]

        for neighbor, moved_index in get_neighbors(current):
            new_path = path + [current]
            heapq.heappush(frontier, (cost + 1 + heuristic(neighbor), cost + 1, neighbor, new_path))
    return None

# Generate human-readable instructions
def generate_instructions(solution_path):
    instructions = []
    for i in range(1, len(solution_path)):
        prev = solution_path[i-1]
        curr = solution_path[i]
        empty_prev = find_empty(prev)
        empty_curr = find_empty(curr)
        moved_ball_color = prev[empty_curr][1]
        instructions.append(f"Step {i}: Move the {moved_ball_color} ball")
    return instructions

# Main program
if __name__ == "__main__":
    initial_state = getInitialState(slots, balls) ### COMMENT OUT TO TEST
    goal_state = [(slot, slot if slot != "white" else "empty") for slot in slots]
    solution_path = a_star(initial_state, goal_state)

    if solution_path:
        print("\nPuzzle Solved! Here's how to do it:")
        instructions = generate_instructions(solution_path)
        for instruction in instructions:
            print(instruction)
    else:
        print("No solution found.")
