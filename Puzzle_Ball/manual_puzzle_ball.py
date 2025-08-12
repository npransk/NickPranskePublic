### DEFINE GAME MECHANICS ###

def getInitialState(slots):
    # Define initial state list
    initial_state = []

    # Loop through and ask user for the initial state of each ball
    for slot in slots:
        while True:  # Repeat until the user provides valid input
            ball_slot = input(f"Which ball is in the {slot} slot?\n")
            if ball_slot not in balls:
                print(f"'{ball_slot}' is not a valid ball color. Please try again.")
            else:
                initial_state.append((slot, ball_slot))
                break  # Exit the inner loop when input is valid
    print("\n------------- GAME START -------------\n")
    return initial_state

def findAvailableMoves(slots, adjacency_map, initial_state):
    # Find where the ball is "Empty"
    empty_slot = next(filter(lambda x: x[1] == "Empty", initial_state), None)[0]
    empty_slot_index = slots.index(empty_slot)

    # Loop through initial_state and list out the available balls to move to the Empty slot
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
    desired_state = [
        ('Black', 'Black'),
        ('Dark Blue', 'Dark Blue'),
        ('Light Blue', 'Light Blue'),
        ('Light Green', 'Light Green'),
        ('Orange', 'Orange'),
        ('Pink', 'Pink'),
        ('Purple', 'Purple'),
        ('Red', 'Red'),
        ('Teal', 'Teal'),
        ('Turquoise', 'Turquoise'),
        ('White', 'Empty'),
        ('Yellow', 'Yellow')
    ]
    return initial_state == desired_state

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
initial_state = getInitialState(slots)

# Option 2: Test version
# initial_state = [
#     ('Black', 'Pink'), ('Dark Blue', 'Black'), ('Light Blue', 'Dark Blue'),
#     ('Light Green', 'Yellow'), ('Orange', 'Turquoise'), ('Pink', 'Purple'),
#     ('Purple', 'Teal'), ('Red', 'Light Blue'), ('Teal', 'Orange'),
#     ('Turquoise', 'Red'), ('White', 'Empty'), ('Yellow', 'Light Green')
# ]

# Option 3: Cheater's version
# initial_state = [
#     ('Black', 'Black'), ('Dark Blue', 'Dark Blue'), ('Light Blue', 'Light Blue'),
#     ('Light Green', 'Light Green'), ('Orange', 'Orange'), ('Pink', 'Pink'),
#     ('Purple', 'Purple'), ('Red', 'Red'), ('Teal', 'Teal'),
#     ('Turquoise', 'Turquoise'), ('White', 'Yellow'), ('Yellow', 'Empty')
# ]

move_count = 0
while not checkForWin(initial_state):
    available_balls = findAvailableMoves(slots, adjacency_map, initial_state)
    initial_state = makeMove(available_balls, initial_state)
    move_count += 1
print(f"Congratulations! You solved the puzzle in {move_count} move(s)")
