
from deuces.deck import Deck
from deuces.card import Card 
from deuces.evaluator import Evaluator 
from deuces.evaluator_with_n_cards import EvaluatorN
from numpy import argmax
from copy import deepcopy
from itertools import combinations_with_replacement, combinations


# Utility function for all_evaluation
def flush_possible(full_board):
    suits = [0,0,0,0]
    suit_conversion = {'d':0, 'c':1, 'h':2, 's':3}
    num_conversion = {0:'d', 1:'c', 2:'h', 3:'s'}
    for card in full_board:
        suits[suit_conversion[Card.int_to_str(card)[1]]] +=1

    suit_amax = argmax(suits)
    max_suits = suits[suit_amax]
    if max_suits >= 3:
        return num_conversion[suit_amax], 5-max_suits
    else:
        return 'n', -1 #flush not possible, null

# Evaluating all possible hands for original hand strength to work
def all_evaluation(full_board, evaluator):
    all_hands = {}
    fp = flush_possible(full_board)
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    suits= ['s', 'c', 'd', 'h']
    equivalent_lists = []
    
    if fp[0] == 'n' or fp[1] ==2:
        for rank in ranks:
            this_rank = []
            for suit in suits:
                card = Card.new(rank+suit)
                if card not in full_board:
                    this_rank.append(card)
            equivalent_lists.append(this_rank)
    elif fp[1] <= 1: #where all cards with the same rank not of the flush suit are indistinct
        altered_suits = deepcopy(suits)
        altered_suits.remove(fp[0])
        for rank in ranks:
            this_rank = []
            for suit in altered_suits:
                card = Card.new(rank+suit)
                if card not in full_board:
                    this_rank.append(card)
            equivalent_lists.append(this_rank) 
            
            card_of_suit = Card.new(rank+fp[0])
            if card_of_suit not in full_board:
                equivalent_lists.append([card_of_suit])

    combos = combinations_with_replacement(equivalent_lists,2)
    
    if fp[1] == 2:
        for combo in combos:
            for equivalent_x in combo[0]:
                for equivalent_y in combo[1]:
                    if equivalent_x != equivalent_y and (Card.int_to_str(equivalent_x)[1] != Card.int_to_str(equivalent_y)[1]):
                        evaluation_most = evaluator.evaluate(full_board, [equivalent_x, equivalent_y])
                        break
                        
            for equivalent_x in combo[0]:
                for equivalent_y in combo[1]:
                    if equivalent_x != equivalent_y and (Card.int_to_str(equivalent_x)[1] != Card.int_to_str(equivalent_y)[1]):
                        all_hands[tuple(sorted([equivalent_x, equivalent_y]))] = evaluation_most
                    elif equivalent_x != equivalent_y and (Card.int_to_str(equivalent_x)[1] == Card.int_to_str(equivalent_y)[1]):
                        special_evaluation = evaluator.evaluate(full_board, [equivalent_x, equivalent_y])
                        all_hands[tuple(sorted([equivalent_x, equivalent_y]))] = special_evaluation
                        
    else:          
        for combo in combos:
            for equivalent_x in combo[0]:
                for equivalent_y in combo[1]:
                    if equivalent_x != equivalent_y:
                        evaluation = evaluator.evaluate(full_board, [equivalent_x, equivalent_y])
                        break
            for equivalent_x in combo[0]:
                for equivalent_y in combo[1]:
                    if equivalent_x != equivalent_y:
                        all_hands[tuple(sorted([equivalent_x, equivalent_y]))] = evaluation
                
    return all_hands

# Original hand strength. Will be replaced with a faster neural network equivalent, but needed for bootstrapping.
def original_hand_strength(our_hand, HS_List): 
    above = 0
    tied = 0
    behind = 0

    ours = tuple(sorted(our_hand))
    our_evaluation = HS_List[ours]
    
    for considered_hand in HS_List.keys():
        first_in_hand = considered_hand[0] in our_hand
        second_in_hand = considered_hand[1] in our_hand
        if (not (first_in_hand or second_in_hand)): #or considered_hand[1] == -1: #We know that certain hands (containing our cards or board cards) are illegal.
            their_evaluation = HS_List[considered_hand]
            if their_evaluation > our_evaluation:
                above += 1
            elif their_evaluation == our_evaluation:
                tied += 1
            else:
                behind += 1
    HS = ((above + tied/2)/(above+tied+behind)) #%hands you win against         
    return HS
