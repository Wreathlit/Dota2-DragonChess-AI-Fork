## Dota2 Dragon Chess Match-3 Game AI

This is an AI for the match-3 minigame embedded in Dota2 Crownfall Act 3. It uses image recognition to identify the elements and performs actions based on specific algorithms.

It will help you beat the game.

### Download
[Here](https://github.com/BurgerNight/Dota2-DragonChess-AI/releases)

### Usage

1. Go to the DragonChess game page.
2. Run `run_agent.exe` run as administrator.
3. Move the mouse over `Play` on game page, then press `b` to start.
4. Watch it play.
5. Use `p` (or `q`, for compatibility) to pause/unpause the program, use `esc` to exit the program.
6. The agent will automatically detect the Dota2 window and calibrate board position.



### Supported Arguments
`--wait_static` or `-w`, if set, it will wait for a static board to perform the next action.

`--show` or `--show_board` or `-s`, if set, it will display the identified game board image, use this for debugging.

`--action_delay`,
Delay after each swap action in seconds. Increase this if pieces have not fully settled before the next move. Default: `0.5`.

`--lookahead_depth`,
Search depth for move planning. Recommended `3~4`. Default: `3`.

`--disable_wait_settle`,
Disable waiting for board settling after each swap.

`--settle_timeout`,
Maximum seconds to wait for board settling after each swap. Default: `2.0`.

You can take a screenshot and paste it to Windows paint to find out the coordinates (as shown below). 
![board_example.png](board_example.png)

### Install Dependencies (Python)
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


### Q&A
1. The program is not functioning properly.

* The program keeps showing
`Cannot locate the game board, make sure the game is on the screen.` even after you switch back to the game.
* The mouse is not clicking within the area of the board or not switching the right element.
* The program raises an error and exits.

These problems occur when the game board is not correctly identified. Try changing game resolution and make sure the Dota2 game window is visible on screen.
These problems occur when the game board is not correctly identified. Try changing game resolution and make sure the Dota2 game window is visible on screen.

2. Will it get you VAC banned?

The program functions based on image recognition. It does not access or modify any game files and should be safe to use.
