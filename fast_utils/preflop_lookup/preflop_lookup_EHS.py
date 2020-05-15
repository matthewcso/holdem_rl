from deuces.card import Card
from deuces.deck import Deck
from fast_utils.hand_strength.original_HS import *
from fast_utils.hand_strength.nn_HS import encode_hs
from fast_utils.expected_hand_strength.nn_EHS import *
from keras.models import load_model

def read_lookup_table(hole_cards, lookup_table):
    sorted_hole = sorted(hole_cards)
    sorted_hole.reverse()
    card_strings = [Card.int_to_str(card) for card in sorted_hole]

    if card_strings[0][1] != card_strings[1][1]:
        suited = False
    else:
        suited = True
    card_strings[0] = card_strings[0][0] + 'd'
    if suited:
        card_strings[1] = card_strings[1][0] +'d'
    else:
        card_strings[1] = card_strings[1][0] +'s'
    card_strings = tuple(card_strings)
    return lookup_table[card_strings]

