# Fast computation of expected hand strength (3 or 4 board cards) with neural network estimation

from fast_utils.hand_strength.original_HS import *
from fast_utils.hand_strength.nn_HS import *
import numpy as np
from scipy.special import comb
from random import sample
from deuces.deck import Deck
from itertools import combinations

def encode_ehs(hole_cards, board, en, HS_model):
    """
    Encodes hole cards, board for EHS prediction.
    Args:
        hole_cards: list of int (deuces cards)
        board: list of int (deuces cards)
        en: EvaluatorN object
        HS_model: keras model (can be hand_strength/HS_model.h5)
    Returns:
        list of numeric: encoding for neural network
    """
    encoded = encode_hs(hole_cards, board, en)

    encoded.append(HS_model.predict(np.asarray([encoded]))[0])
    return encoded

def original_EHS(hole_cards, board, en, model,  max_n = 1200, fn = 'auto'): 
    """
    Function used to generate EHS for training the EHS model.
    Args:
        hole_cards: list of int (deuces cards)
        board: list of int (deuces cards)
        en: EvaluatorN object
        model: keras model (can be hand_strength/HS_model.h5) for hand strength
        max_n: maximum number of combinations to consider
        fn: if 'auto', fill the board such that there are 5 cards. Otherwise, draw that many cards. 
    Returns:
        tuple (float, float): EHS, EHS^2

    """
    sample_deck = Deck()

    sample_deck.remove(board+hole_cards)
    
    if fn == 'auto':
        fill_num = 5 - len(board)
    else:
        fill_num = fn

    if fill_num == 0:
        features = np.asarray([encode_hs(hole_cards, board, en)]) 
    else:
        if comb(len(sample_deck.cards), fill_num) <= max_n:
            next_possibilities = combinations(sample_deck.cards, fill_num) 
        else:
            next_possibilities = [sample(sample_deck.cards, fill_num) for _ in range(max_n)]

        features = np.asarray([encode_hs(hole_cards, board+list(n), en) for n in next_possibilities])

    HSs = model.predict(features)
    
    this_EHS = np.mean(HSs)
    this_EHSS = np.mean(HSs**2)

    return this_EHS, this_EHSS

def nn_EHS(hole_cards, board, en, HS_model, model):
    """
    Predict the EHS using neural network.
    Args:
        hole_cards: list of int (deuces cards)
        board: list of int (deuces cards)
        en: EvaluatorN object
        HS_model: keras model (can be hand_strength/HS_model.h5) for hand strength
        model: keras model (can be expected_hand_strength/EHS_model.h5) for expected hand strength
    Returns:
        float: EHS
    """
    features = np.asarray([encode_ehs(hole_cards, board, en, HS_model)]) 

    this_HS = model.predict(features)
    return this_HS[0][0]