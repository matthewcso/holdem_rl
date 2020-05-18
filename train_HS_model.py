# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### To speed up the really common HS calculation, we are going to train a neural network to do it. 

# %%
from fast_utils.hand_strength.original_HS import *
from fast_utils.hand_strength.nn_HS import encode_hs
from sklearn.model_selection import train_test_split
from pickle import dump, load
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from random import choice


# %%
try: 
    encodings = load(open('fast_utils/hand_strength/original_HS_training_x.pickle','rb'))
    HSs = load(open('fast_utils/hand_strength/original_HS_training_y.pickle','rb'))
except FileNotFoundError:
    d = Deck()
    en = EvaluatorN()

    encodings = []
    HSs = []

    for board_i in range(10000):
        d.shuffle()
        board = d.draw(choice([3,4,5]))
        all_evals = all_evaluation(board, en)

        for hand_i in range(100):
            d.shuffle()
            d.remove(board)
            our_hand = d.draw(2)
            encodings.append(encode_hs(our_hand, board, en))
            HSs.append(original_hand_strength(our_hand, all_evals))
        if board_i % 100 == 0:
            print(board_i)
    dump(encodings, open('fast_utils/hand_strength/original_HS_training_x.pickle','wb'))
    dump(HSs, open('fast_utils/hand_strength/original_HS_training_y.pickle','wb'))


# %%
x_train, x_test, y_train, y_test = train_test_split(np.asarray(encodings), np.asarray(HSs), test_size=0.05)


# %%
model = Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)


# %%
model.save("fast_utils/hand_strength/HS_model.h5")


