import heapq

def getCurrentState(slots):
    # Define current state dict
    current_state = []

    # Loop through and ask user for the current state of each ball
    for slot in slots:
        while True:  # Repeat until the user provides valid input
            ball_slot = input(f"Which ball is in the {slot} slot?\n")
            if ball_slot not in balls:
                print(f"'{ball_slot}' is not a valid ball color. Please try again.")
            else:
                current_state = (slot, ball_slot)
                break  # Exit the inner loop when input is valid
    
    return current_state

def heuristic(state, goal_state):
    # Count the number of misplaced balls (excluding the empty slot)
    return sum(1 for (slot, ball), (g_slot, g_ball) in zip(state, goal_state) if ball != g_ball and ball != "empty")

def get_neighbors(state, slots, adjacency_map):
    empty_slot = next(filter(lambda x: x[1] == "empty", state))[0]
    empty_index = slots.index(empty_slot)
    
    neighbors = []
    for adj_index in adjacency_map[empty_index]:
        adj_slot = slots[adj_index]
        adj_ball = next(b for s, b in state if s == adj_slot)
        
        # Generate new state by swapping empty slot with adjacent ball
        new_state = [list(pair) for pair in state]
        for pair in new_state:
            if pair[0] == empty_slot:
                pair[1] = adj_ball
            elif pair[0] == adj_slot:
                pair[1] = "empty"
        new_state = tuple(map(tuple, new_state))
        
        neighbors.append((new_state, adj_ball))
    
    return neighbors

def a_star_solver(start_state, goal_state, slots, adjacency_map):
    open_set = []
    heapq.heappush(open_set, (0, start_state, []))  # (priority, state, path)
    came_from = {}
    g_score = {start_state: 0}
    
    while open_set:
        _, current, path = heapq.heappop(open_set)
        
        if current == goal_state:
            return path  # Return the sequence of moves to solve the puzzle
        
        for neighbor, moved_ball in get_neighbors(current, slots, adjacency_map):
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal_state)
                heapq.heappush(open_set, (f_score, neighbor, path + [moved_ball]))
                came_from[neighbor] = current
    
    return None  # No solution found

# Define the slots in a list in alphabetical order (although the order doesn't matter)
slots = ["black",       # 0
         "dark_blue",   # 1
         "light_blue",  # 2
         "light_green", # 3
         "orange",      # 4
         "pink",        # 5
         "purple",      # 6
         "red",         # 7
         "teal",        # 8
         "turquoise",   # 9
         "white",       # 10
         "yellow"       # 11
]

# Copy slots list to balls and change white to empty
balls = slots[:]
balls[balls.index("white")] = "empty"

# Define which slots are surrounded by which other slots
adjacency_map = {
    0: [1,5,6,8,11],    # black: dark_blue, pink, purple, teal, yellow
    1: [0,2,5,8,9],     # dark_blue: black, light_blue, pink, teal, turquoise
    2: [1,5,7,9,10],    # light_blue: dark_blue, pink, red, turquoise, white
    3: [4,6,7,10,11],   # light_green: orange, purple, red, white, yellow
    4: [3,6,7,8,9],     # orange: light_green, purple, red, teal, turquoise
    5: [0,1,2,10,11],   # pink: black, dark_blue, light_blue, white, yellow
    6: [0,3,4,8,11],    # purple: black, light_green, orange, teal, yellow
    7: [2,3,4,9,10],    # red: light_blue, light_green, orange, turquoise, white
    8: [0,1,4,6,9],     # teal: black, dark_blue, orange, purple, turquoise
    9: [1,2,4,7,8],     # turquoise: dark_blue, light_blue, orange, red, teal
    10: [2,3,5,7,11],   # white: light_blue, light_green, pink, red, yellow
    11: [0,3,5,6,10]    # yellow: black, light_green, pink, purple, white
}

# Define goal state
goal_state = (
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
    ('white', 'empty'),
    ('yellow', 'yellow')
)

# ## TEST VERSION
# current_state = [
#     ('black', 'pink'), 
#     ('dark_blue', 'black'), 
#     ('light_blue', 'dark_blue'), 
#     ('light_green', 'yellow'), 
#     ('orange', 'turquoise'), 
#     ('pink', 'purple'), 
#     ('purple', 'teal'), 
#     ('red', 'light_blue'), 
#     ('teal', 'orange'), 
#     ('turquoise', 'red'), 
#     ('white', 'empty'), 
#     ('yellow', 'light_green')
#     ]

## CHEATER'S VERSION (move yellow to win)
# current_state = [
#     ('black', 'black'), 
#     ('dark_blue', 'dark_blue'), 
#     ('light_blue', 'light_blue'), 
#     ('light_green', 'light_green'), 
#     ('orange', 'orange'), 
#     ('pink', 'pink'), 
#     ('purple', 'purple'), 
#     ('red', 'red'), 
#     ('teal', 'teal'), 
#     ('turquoise', 'turquoise'), 
#     ('white', 'yellow'), 
#     ('yellow', 'empty')
#     ]


# Convert current state to tuple format
start_state = tuple(getCurrentState(slots))

# start_state = tuple(current_state)

solution = a_star_solver(start_state, goal_state, slots, adjacency_map)

if solution:
    print("Steps to solve the puzzle:")
    for step, move in enumerate(solution, 1):
        print(f"Step {step}: Move {move} to the empty slot")
else:
    print("No solution found.")
