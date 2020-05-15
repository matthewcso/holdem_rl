
from fast_utils.hand_strength.original_HS import *
from fast_utils.hand_strength.nn_HS import *
from fast_utils.expected_hand_strength.nn_EHS import *
from fast_utils.preflop_lookup.preflop_lookup_EHS import *
import numpy as np

def unified_EHS(hole_cards, board, en, HS_model, EHS_model, lookup):
    if len(board) == 0:
        return read_lookup_table(hole_cards, lookup)[0]
    elif len(board) == 5:
        return nn_HS(hole_cards, board, en, HS_model)
    else:
        return nn_EHS(hole_cards, board, en, HS_model, EHS_model)