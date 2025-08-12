import heapq
from collections import deque

### DEFINE GAME MECHANICS ###

def getInitialState(slots, balls):
    initial_state = []
    for slot in slots:
        while True:
            ball_slot = input(f"Which ball is in the {slot} slot?\n")
            if ball_slot not in balls:
                print(f"'{ball_slot}' is not a valid ball color. Please try again.")
            else:
                initial_state.append((slot, ball_slot))
                break
    print("\n------------- GAME START -------------\n")
    return initial_state

def findAvailableMoves(slots, adjacency_map, initial_state):
    empty_slot = next(filter(lambda x: x[1] == "Empty", initial_state), None)[0]
    empty_slot_index = slots.index(empty_slot)
    available_balls = []
    for adjacent_slot_index in adjacency_map[empty_slot_index]:
        adjacent_slot_color = slots[adjacent_slot_index]
        adjacent_ball_color = next(b for s, b in initial_state if s == adjacent_slot_color)
        available_balls.append(adjacent_ball_color)
    return available_balls

def makeMove(available_balls, initial_state):
    while True:
        selected_ball = input(f"\nWhich ball would you like to move?:\n{chr(10).join(available_balls)}\n\n")
        if selected_ball not in available_balls:
            print(f"{selected_ball} is not an available ball. Please try again")
        else:
            break
    empty_index = next(i for i, (s, b) in enumerate(initial_state) if b == "Empty")
    selected_index = next(i for i, (s, b) in enumerate(initial_state) if b == selected_ball)
    initial_state[empty_index] = (initial_state[empty_index][0], selected_ball)
    initial_state[selected_index] = (initial_state[selected_index][0], "Empty")
    return initial_state

def checkForWin(initial_state):
    desired_state = [(slot, slot if slot != "White" else "Empty") for slot in slots]
    return initial_state == desired_state

### A* ALGORITHM FUNCTIONS ###

def computeDistances(adjacency_map):
    distances = {}
    for start in adjacency_map:
        distances[start] = {}
        queue = deque([(start, 0)])
        visited = set()
        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            distances[start][node] = dist
            for neighbor in adjacency_map[node]:
                queue.append((neighbor, dist + 1))
    return distances

def heuristic(state, goal, distances):
    total = 0
    for i, (slot, ball) in enumerate(state):
        if ball.lower() == "empty" or ball == slot:
            continue
        goal_index = next(j for j, (s, _) in enumerate(goal) if s == ball)
        total += distances[i][goal_index]
    return total

def findEmpty(state):
    for i, (_, ball) in enumerate(state):
        if ball.lower() == "empty":
            return i
    return -1

def getNeighbors(state):
    empty_index = findEmpty(state)
    neighbors = []
    for adj in adjacency_map[empty_index]:
        new_state = state.copy()
        new_state[empty_index], new_state[adj] = (new_state[empty_index][0], new_state[adj][1]), (new_state[adj][0], "Empty")
        neighbors.append((new_state, adj))
    return neighbors

def aStar(start, goal, distances):
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal, distances), 0, start, []))
    visited = set()
    while frontier:
        _, cost, current, path = heapq.heappop(frontier)
        state_tuple = tuple(ball for _, ball in current)
        if state_tuple in visited:
            continue
        visited.add(state_tuple)
        if current == goal:
            return path + [current]
        for neighbor, moved_index in getNeighbors(current):
            new_path = path + [current]
            heapq.heappush(frontier, (cost + 1 + heuristic(neighbor, goal, distances), cost + 1, neighbor, new_path))
    return None

### DEFINE GAME PARAMETERS ###
slots = [
    "Black", "Dark Blue", "Light Blue", "Light Green", "Orange", "Pink",
    "Purple", "Red", "Teal", "Turquoise", "White", "Yellow"
]

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

balls = slots[:]
balls[balls.index("White")] = "Empty"

### PLAY GAME ###

# Uncomment one of the following to test

# Option 1: User input
# initial_state = getInitialState(slots)

# Option 2: Test version
# initial_state = [
#     ('Black', 'Pink'), ('Dark Blue', 'Black'), ('Light Blue', 'Dark Blue'),
#     ('Light Green', 'Yellow'), ('Orange', 'Turquoise'), ('Pink', 'Purple'),
#     ('Purple', 'Teal'), ('Red', 'Light Blue'), ('Teal', 'Orange'),
#     ('Turquoise', 'Red'), ('White', 'Empty'), ('Yellow', 'Light Green')
# ]

# Option 3: Cheater's version
initial_state = [
    ('Black', 'Black'), ('Dark Blue', 'Dark Blue'), ('Light Blue', 'Light Blue'),
    ('Light Green', 'Light Green'), ('Orange', 'Orange'), ('Pink', 'Pink'),
    ('Purple', 'Purple'), ('Red', 'Red'), ('Teal', 'Teal'),
    ('Turquoise', 'Turquoise'), ('White', 'Yellow'), ('Yellow', 'Empty')
]

# Calculate optimal moves using A* algorithm
goal_state = [(slot, slot if slot != "White" else "Empty") for slot in slots]
distances = computeDistances(adjacency_map)
solution_path = aStar(initial_state, goal_state, distances)
optimal_moves = len(solution_path) - 1 if solution_path else "unknown"

# Play game 
move_count = 0
while not checkForWin(initial_state):
    available_balls = findAvailableMoves(slots, adjacency_map, initial_state)
    initial_state = makeMove(available_balls, initial_state)
    move_count += 1

print(f"Congratulations! You solved the puzzle in {move_count} move(s). The optimal number of moves was {optimal_moves}.")
