# Ludo Arena

Playable four-player Ludo app for Nick vs Codex.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Run those commands from this folder.

## Board Indexing

Each piece is stored as a position relative to its own start square:

- `-1`: in the starting circle
- `0..51`: on the shared 52-space track
- `52..56`: in that player's five-space home column
- `57`: in the home triangle

Absolute track positions are only used while a piece is on the shared track:

```python
absolute = (PLAYER_SPECS[player].start_index + relative_position) % 52
```

The start squares are:

- Red: absolute `0`
- Yellow: absolute `13`
- Blue: absolute `26`
- Green: absolute `39`

Those four absolute squares are globally safe. Pieces in home columns are also safe because opponents cannot enter them.

## Bot Move Contract

Edit [bots.py](bots.py) to replace Yellow and Blue. Your bot method receives:

```python
def choose_move(self, state, legal_moves, rng) -> int:
    ...
```

Return the 0-based piece index you want to move:

- `0` means the player's first piece, such as `Y1`
- `1` means the second piece
- `2` means the third piece
- `3` means the fourth piece

Only return a piece index that appears in `legal_moves`. Each `MoveOption` includes:

- `piece_index`
- `source`
- `target`
- `target_absolute`
- `bumps`
- `is_from_base`
- `reaches_home_triangle`
- `enters_home_column`
- `lands_on_safe_square`

Example:

```python
class YellowProbabilityBot:
    name = "Nick probability bot"

    def choose_move(self, state, legal_moves, rng) -> int:
        best_move = max(legal_moves, key=lambda move: (len(move.bumps), move.target))
        return best_move.piece_index
```

The app validates bot output. If a bot raises an exception or returns an illegal piece index, the app uses a legal fallback so the game can continue.

## Debugging Bots

The sidebar has active-player toggles. Choose the players you want, then start a new game with that selection.

The **Simulation Lab** tab runs automated batches using the active player selection. Red uses a neutral fallback policy during simulations because Red is manual during normal play. The output includes:

- wins and win percentage
- rolls, moves, passes, and extra rolls
- how many times each player bumped someone
- how many times each player got bumped
- three-6 turn losses
- base exits, home entries, finished pieces, and total pips moved

## Tests

```powershell
python -m unittest discover -s tests
```
