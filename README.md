# holdem_rl
Improvement of RLCard holdem env, fast neural network approximation of expected hand strength. Original RLCard: https://github.com/datamllab/rlcard 
Additionally, quick port of the python2 library deuces to python3. https://github.com/worldveil/deuces 

rlcard/envs/nolimitholdem and rlcard/games/nolimitholdem are modified to have a better one-hot encoding of board, expected hand strength, pot size, and more.

Expected hand strength is a common way to group hand strengths in computer poker (although a bit outdated). By using a neural network to approximate the Monte Carlo simulation, we can speed this process up significantly. To retrain the models, run train_EHS_model.py and train_HS_model.py.

Don't expect things in unused to be maintained.

