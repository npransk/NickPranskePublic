import heapq
from collections import deque

# --- Adjacency Map ---
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

# --- Slot and Ball Definitions ---
slots = ["Black", "Dark Blue", "Light Blue", "Light Green", "Orange", "Pink",
         "Purple", "Red", "Teal", "Turquoise", "White", "Yellow"]

balls = ["Black", "Dark Blue", "Light Blue", "Light Green", "Orange", "Pink",
         "Purple", "Red", "Teal", "Turquoise", "Yellow", "Empty"]

# --- Input Function ---
def getInitialState(slots, balls):
    initial_state = []
    print("Please enter the ball color for each slot. Valid options are:")
    print(", ".join(balls))
    for slot in slots:
        while True:
            ball_slot = input(f"Which ball is in the '{slot}' slot? ")
            if ball_slot not in balls:
                print(f"'{ball_slot}' is not a valid ball color. Please try again.")
            else:
                initial_state.append((slot, ball_slot))
                break
    return initial_state

# --- Distance Computation ---
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

# --- Heuristic ---
def heuristic(state, goal, distances):
    total = 0
    for i, (slot, ball) in enumerate(state):
        if ball.lower() == "empty" or ball == slot:
            continue
        goal_index = next(j for j, (s, _) in enumerate(goal) if s == ball)
        total += distances[i][goal_index]
    return total

# --- Find Empty Slot ---
def findEmpty(state):
    for i, (_, ball) in enumerate(state):
        if ball.lower() == "empty":
            return i
    return -1

# --- Generate Neighbors ---
def getNeighbors(state):
    empty_index = findEmpty(state)
    neighbors = []
    for adj in adjacency_map[empty_index]:
        new_state = state.copy()
        new_state[empty_index], new_state[adj] = (new_state[empty_index][0], new_state[adj][1]), (new_state[adj][0], "Empty")
        neighbors.append((new_state, adj))
    return neighbors
# --- A* Search ---
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

# --- Instructions ---
def generateInstructions(solution_path):
    instructions = []
    for i in range(1, len(solution_path)):
        prev = solution_path[i-1]
        curr = solution_path[i]
        empty_prev = findEmpty(prev)
        empty_curr = findEmpty(curr)
        moved_ball_color = prev[empty_curr][1]
        instructions.append(f"Step {i}: Move the {moved_ball_color} ball")
    return instructions

# --- Main Execution ---
if __name__ == "__main__":
    initial_state = getInitialState(slots, balls)
    goal_state = [(slot, slot if slot != "White" else "Empty") for slot in slots]
    distances = computeDistances(adjacency_map)
    solution_path = aStar(initial_state, goal_state, distances)

    if solution_path:
        print("\nPuzzle Solved! Here's how to do it:")
        instructions = generateInstructions(solution_path)
        for instruction in instructions:
            print(instruction)
    else:
        print("No solution found. Please check your input.")
