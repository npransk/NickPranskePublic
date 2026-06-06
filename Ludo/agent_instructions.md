# Ludo Playable Game/AI Bot
## By Nick Pranske

### Project Overview
I want to create a playable python-based Ludo game with 4 players, each with its own persona. The first player is a human-playable player. The second will be developed completely by me without the help of AI. It will use a probability-based strategy that I will define. The third player is a reinforcement learning bot that I will also create from scratch. The fourth will be created by you and you have complete creative freedom. Your goal will be to make a better bot than me. You only have one-shot though, so make sure to do all your design, thought-process, documentation, training, testing, or any other processes all in the same prompt loop. If you need GPU resources, you can ask me to deploy the script in a Google Colab notebook or an equivalent in order to get the necessary resources - but you have to define it. This is me against you.

### Allowed Processes
You are allowed to:
- Have access and make changes only to this repository and only the Ludo/ folder.
- Run multiple iterations within the same session, but not multiple sessions.
- Access research resources on the internet

### Starting Tasks
In order to create the playing field, I need you to design an interactive Ludo app (I've used Streamlit before, but use whatever you want). The game rules are defined in the section of this document titled, **Game Rules**. Use the classic Ludo colors of Red (bottom left), Yellow (top left), Blue (top right), and Green (bottom right). Red will be the manual player and you will write the code for that. Yellow will be my probability bot, Blue will be my RL bot, and Green will be yours. Leave places in the code for me to add my 2 players' code and explain to me how I should output my moves, according to your code scaffolding

### Game Rules
These rules are an adaptation of the rules on https://www.mastersofgames.com/rules/ludo-rules-instructions-guide.htm, but I have purposefully excluded "blocks" in this game. I have also included my own roll of three 6s in a row.

1. Players take turns in a clockwise order. For this game, Red will always start (for better user interface)
2. Each throw, the player decides which piece to move. A piece simply moves in a clockwise direction around the track given by the number thrown. If no piece can legally move according to the number thrown, play passes to the next player. For example, if the only piece on the board for a player is 3 spots away from the home triangle, but the player rolls a 4, play passes to the next player. If the player had pieces in the starting circle, then a 6 would be legal to move the piece from the starting circle
3. A throw of 6 gives another turn. The player can move any piece, not just the piece previously moved.
4. Three 6s in a row results in a loss of turn and the third 6 does not move the piece. A "bump" (landing on another player) resets the counter of 6s. The movement from the first 2 6s remains.
5. A player must throw a 6 to move a piece from the starting circle onto the first square on the track. The first 6 only lets the piece exit the starting circle - it does not move, but the player does bet another roll
6. If a piece lands on a piece of a different colour, the piece landed on is returned to its starting circle.
7. The four starting squares are globally safe squares. Any piece occupying one of those squares cannot be bumped regardless of color.
8. There is no limit on how many pieces can be on the same safe spot (multiple players' pieces can all be on it) and any player can stack its own pieces on the same spot safely. If another player's piece lands on it, all pieces on that spot are bumped.
9. Each player's final 5 spots (home column) are safe and other players cannot enter them.

When a piece has circumnavigated the board, it proceeds up the home column. A piece can only be moved onto the home triangle by an exact throw. The first player to move all 4 pieces into the home triangle wins.

### Logistic Details
- There are 52 spaces on a classic Ludo board, not including each player's 5 home column spots and the 1 home triangle. In order for us to do math on the board, we need a dual-indexed system - one for absolute position and one for position relative to each player's start. You can design the indexing and explain to me how it works
- It would be cool to have an animation in the app for each piece's moves, but it's not a necessary detail.
- We are playing with only 1 die - I know some version do 2 but we aren't
- As stated, blocks don't exist in this version


