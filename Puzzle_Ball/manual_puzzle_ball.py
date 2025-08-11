### DEFINE GAME MECHANICS ###

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

def findAvailableMoves(slots, adjacency_map, current_state):
    # Find where the ball is "empty"
    empty_slot = next(filter(lambda x: x[1] == "empty", current_state),None)[0]
    empty_slot_index = slots.index(empty_slot)

    # Loop through current_state and list out the available balls to move to the empty slot
    available_balls = []
    for adjacent_slot_index in adjacency_map[empty_slot_index]:
        adjacent_slot_color = slots[adjacent_slot_index]
        adjacent_ball_color = next(b for s, b in current_state if s == adjacent_slot_color)
        available_balls.append(adjacent_ball_color)
    
    return available_balls

def makeMove(available_balls, current_state):
    while True:
        selected_ball = input(f"Which ball would you like to move to the empty slot?:\n{chr(10).join([ball for ball in available_balls])}\n\n")
        if selected_ball not in available_balls:
            print(f"{selected_ball} is not an available ball. Please try again")
        else:
            break

    empty_index_current_state = next(i for i, (s, b) in enumerate(current_state) if b == "empty")
    selected_ball_index_current_state = next(i for i, (s, b) in enumerate(current_state) if b == selected_ball)
    current_state[empty_index_current_state] = (current_state[empty_index_current_state][0], selected_ball)
    current_state[selected_ball_index_current_state] = (current_state[selected_ball_index_current_state][0], "empty")

    return current_state

def checkForWin(current_state):
    desired_state = [
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
    ]

    result = current_state == desired_state
    return result

### DEFINE GAME PARAMETERS ###

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

# Copy slots list to balls and change white to empty
balls = slots[:]
balls[balls.index("white")] = "empty"

# Have user input custom current state (by shuffling the physical puzzle) or using the default
# current_state = getCurrentState(slots)
# print(current_state)

## TEST VERSION

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
current_state = [
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


### PLAY GAME ###
move_count = 0
while checkForWin(current_state) == False:
    available_balls = findAvailableMoves(slots, adjacency_map, current_state)
    current_state = makeMove(available_balls, current_state)
    move_count+=1

print(f"Congratulations! You solved the puzzle in {move_count} move(s)")

