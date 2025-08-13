import streamlit as st
import heapq
from collections import deque

# --- Puzzle Setup ---
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

slots = ["Black", "Dark Blue", "Light Blue", "Light Green", "Orange", "Pink",
         "Purple", "Red", "Teal", "Turquoise", "White", "Yellow"]

balls = ["Black", "Dark Blue", "Light Blue", "Light Green", "Orange", "Pink",
         "Purple", "Red", "Teal", "Turquoise", "Yellow", "Empty"]

# --- Solver Functions ---
def compute_distances(adjacency_map):
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
        if ball == "empty" or ball == slot:
            continue
        goal_index = next(j for j, (s, _) in enumerate(goal) if s == ball)
        total += distances[i][goal_index]
    return total

def find_empty(state):
    for i, (_, ball) in enumerate(state):
        if ball == "empty":
            return i
    return -1

def get_neighbors(state):
    empty_index = find_empty(state)
    neighbors = []
    for adj in adjacency_map[empty_index]:
        new_state = state.copy()
        new_state[empty_index], new_state[adj] = (new_state[empty_index][0], new_state[adj][1]), (new_state[adj][0], "empty")
        neighbors.append((new_state, adj))
    return neighbors

def a_star(start, goal, distances):
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

        for neighbor, moved_index in get_neighbors(current):
            new_path = path + [current]
            heapq.heappush(frontier, (cost + 1 + heuristic(neighbor, goal, distances), cost + 1, neighbor, new_path))
    return None

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

# --- Streamlit UI ---
st.title("Nick Pranske's Puzzle Ball Solver")
st.markdown("Select the current ball color for each slot:")

user_input = []
for slot in slots:
    default_index = balls.index(slot) if slot in balls else 0
    selected_ball = st.selectbox(
        f"{slot} slot",
        balls,
        index=default_index,
        key=slot
    )
    user_input.append((slot, selected_ball))

if st.button("Solve"):
    goal_state = [(slot, slot if slot != "white" else "empty") for slot in slots]
    distances = compute_distances(adjacency_map)
    solution_path = a_star(user_input, goal_state, distances)

    if solution_path:
        st.success("Puzzle Solved! Here's how to do it:")
        instructions = generate_instructions(solution_path)
        for instruction in instructions:
            st.write(instruction)
    else:
        st.error("No solution found. Please check your input.")
