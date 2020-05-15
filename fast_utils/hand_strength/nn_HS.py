# Fast computation of hand strength with neural network estimation

from fast_utils.hand_strength.original_HS import *
import numpy as np

def encode_hs(hole_cards, board, en):
    suit, necessary = flush_possible(board)
    evaluated, used_cards = en.evaluate(hole_cards, board)
    n_used = 0
    for i in used_cards:
        if i in hole_cards:
            n_used += 1

    rank_encoder = {'A':14, 'K':13, 'Q':12, 'J':11, 'T':10, '9':9, '8':8, '7':7, '6': 6, '5':5, '4':4, '3':3,'2':2}
    suit_encoder = {'d':0, 'c':1, 'h':2, 's':3}

    encoded_hole_r = [0 for _ in range(13)]
    encoded_hole_s = [0 for _ in range(4)]
    encoded_board_r = [0 for _ in range(13)]
    encoded_board_s = [0 for _ in range(4)]

    for i, z in enumerate(hole_cards):
        card_str = Card.int_to_str(z)
        encoded_hole_r[rank_encoder[card_str[0]]-2] += 1
        encoded_hole_s[suit_encoder[card_str[1]]] += 1
    
    for i, z in enumerate(board):
        card_str = Card.int_to_str(z)
        encoded_board_r[rank_encoder[card_str[0]]-2] += 1
        encoded_board_s[suit_encoder[card_str[1]]] += 1

    n_flush = sum([Card.int_to_str(x)[1] ==suit for x in hole_cards])
    return encoded_hole_r + encoded_hole_s + encoded_board_r + encoded_board_s + [necessary] + [n_flush] + [n_used]+[evaluated/10000]

def nn_HS(hole_cards, board, en, model):
    features = np.asarray([encode_hs(hole_cards, board, en)]) 
    this_HS = model.predict(features)

    return this_HS[0][0]