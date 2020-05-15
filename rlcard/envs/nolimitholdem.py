import json
import os
import numpy as np

import rlcard
from rlcard.envs import Env
from rlcard.games.nolimitholdem import Game
from rlcard.games.nolimitholdem.round import Action

from fast_utils.important_imports import *
from better_features.encode_multihot import *
from better_features.conversion import to_deuces_intlist

from deuces.evaluator_with_n_cards import EvaluatorN
en = EvaluatorN()

class NolimitholdemEnv(Env):
    ''' Limitholdem Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.game = Game()
        super().__init__(config)
        self.actions = Action
        self.state_shape = [125]#54]
        # for raise_amount in range(1, self.game.init_chips+1):
        #     self.actions.append(raise_amount)

        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        #  Stuff we would like in our observation:
        # General features
        #  - call size as a %pot DONE
        #  - all in size as a %pot DONE WITH ISSUES 
        #  - number of others in hand - DONE
        #  - number of others in hand who have raised or called before in hand - DONE 
        #  - number of others in hand who need to call the current raise - DONE 
        #  - street number - DONE 
        #  - board position - DONE 

        # Hand features 
        # Better encoded board and player cards DONE
        # preflop EHS of card pairs vs all (and vs premium) - DONE 
        # postflop (showdown) EHS vs all and vs premium - DONE
        #
        # History features
        # aggression on each street - DONE WITH PAST + CURRENT
        # 

        # print(state)

        # Better encodings
        encoded_public_cards = encode_multihot(state['public_cards'])
        encoded_private_cards = encode_multihot(state['hand'])
        call_percent = state['to_call']
        all_in_percent = state['to_allin']
        n_others = state['n_others']
        position = state['position']
        already_called = state['already_called']
        need_to_call = state['need_to_call']
        pot = state['pot']
        EHS_preflop = unified_EHS(to_deuces_intlist(state['hand']), [], en, hs_model, ehs_model, lookup_table) 
        EHS_postflop = unified_EHS(to_deuces_intlist(state['hand']), to_deuces_intlist(state['public_cards']), en, hs_model, ehs_model, lookup_table) 
        past_aggression = state['past_aggression']
        street_aggression=state['street_aggression']
        my_chips = state['my_chips']
        all_chips = state['all_chips']

        obs = encoded_public_cards[0].tolist() + encoded_public_cards[1].tolist()+encoded_private_cards[0].tolist() + encoded_public_cards[1].tolist()+[call_percent,all_in_percent, n_others,position,already_called,need_to_call,pot,EHS_preflop,EHS_postflop,past_aggression, street_aggression,float(my_chips),float(max(all_chips))]
        obs = np.asarray(obs) #shape (125,)

        extracted_state = {}

        legal_actions = [action.value for action in state['legal_actions']]
        extracted_state['legal_actions'] = legal_actions

     #   public_cards = state['public_cards']
     #   hand = state['hand']
     #   my_chips = state['my_chips']
     #   all_chips = state['all_chips']
     #   cards = public_cards + hand

     #   idx = [self.card2index[card] for card in cards]
     #   obs = np.zeros(self.state_shape[0]) 
     #   obs[idx] = 1
     #   obs[52] = float(my_chips)
     #   obs[53] = float(max(all_chips))
        extracted_state['obs'] = obs

        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions(action_id) not in legal_actions:
            if Action.CHECK in legal_actions:
                return Action.CHECK
            else:
                print("Tried non legal action", action_id, self.actions(action_id), legal_actions)
                return Action.FOLD
        return self.actions(action_id)

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.player_num)]
        state['public_card'] = [c.get_index() for c in self.game.public_cards] if self.game.public_cards else None
        state['hand_cards'] = [[c.get_index() for c in self.game.players[i].hand] for i in range(self.player_num)]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state


