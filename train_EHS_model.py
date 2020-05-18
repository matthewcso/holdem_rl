# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### To speed up the really common HS calculation, we are going to train a neural network to do it. 

# %%
from fast_utils.hand_strength.original_HS import *
from fast_utils.hand_strength.nn_HS import encode_hs
from fast_utils.expected_hand_strength.nn_EHS import *

from sklearn.model_selection import train_test_split
from pickle import dump, load
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
from random import choice
from copy import deepcopy
from deuces.deck import Deck
from deuces.evaluator import Evaluator


# %%
hs_model = load_model('fast_utils/hand_strength/HS_model.h5')


# %%
try: 
    encodings = load(open('fast_utils/expected_hand_strength/original_EHS_training_x.pickle','rb'))
    EHSs = load(open('fast_utils/expected_hand_strength/original_EHS_training_y.pickle','rb'))
except FileNotFoundError:
    d = Deck()
    en = EvaluatorN()

    encodings = []
    EHSs = []

    for board_i in range(150000):
        d.shuffle()
        our_hand = d.draw(2)
        board = d.draw(choice([3,4]))
        encodings.append(encode_ehs(our_hand, board, en, hs_model))
        EHSs.append(original_EHS(our_hand, board, en, hs_model, 100))
        if board_i % 100 == 0:
            print(board_i)
    dump(encodings, open('fast_utils/expected_hand_strength/original_EHS_training_x.pickle','wb'))
    dump(HSs, open('fast_utils/expected_hand_strength/original_EHS_training_y.pickle','wb'))


# %%
x_train, x_test, y_train, y_test = train_test_split(np.asarray(encodings[:-1]), np.asarray(EHSs)[:,0], test_size=0.05)


# %%
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
model.save("fast_utils/expected_hand_strength/EHS_model.h5")


# %%
d = Deck()
ours = d.draw(2)
board = d.draw(3)
print(nn_EHS(ours, board, en, hs_model, model))
print(original_EHS(ours, board, en, hs_model))


