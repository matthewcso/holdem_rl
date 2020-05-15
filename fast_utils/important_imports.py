from deuces.card import Card
from deuces.evaluator_with_n_cards import EvaluatorN
from fast_utils.unified_EHS import unified_EHS
from keras.models import load_model

hs_model = load_model('fast_utils/hand_strength/HS_model.h5')
ehs_model = load_model('fast_utils/expected_hand_strength/EHS_model.h5')
with open('fast_utils/preflop_lookup/preflop_EHSs.txt', 'r') as lookup:
    lookup_table = eval(lookup.read())
en = EvaluatorN