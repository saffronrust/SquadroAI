# SquadroAI
An AI for the board game Squadro using Monte Carlo Tree Search and reinforcement learning.

## Setup
Run `pip install requirements.txt` to install the necessary dependencies.

## Training the AI
Run `python train.py`, then choose the number of games you want to train it on, a good number would be 500, but you can choose any number you like.

## Logging the Training
After running `python train.py`, create a separate terminal and run `tensorboard --logdir=runs`, and open up `http://localhost:6006/` in your browser, you'll be able to see the losses and win rate as training progresses.

## How to Play
Run `python main.py`, and select what game mode you would like.
1. Human vs AI
2. Human vs Human
3. AI vs AI

Have fun!