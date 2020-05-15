

import itertools
from .card import Card
from .deck import Deck
from .lookup import LookupTable
from .evaluator import Evaluator

class EvaluatorN(Evaluator):#modified Evaluator to return hole cards used to make the strongest hand

    def _five(self, cards):
        """
        Performs an evalution given cards in integer form, mapping them to
        a rank in the range [1, 7462], with lower ranks being more powerful.
        Variant of Cactus Kev's 5 card evaluator, though I saved a lot of memory
        space using a hash table and condensing some of the calculations. 
        """
        # if flush
        if cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000:
            handOR = (cards[0] | cards[1] | cards[2] | cards[3] | cards[4]) >> 16
            prime = Card.prime_product_from_rankbits(handOR)
            return self.table.flush_lookup[prime], cards

        # otherwise
        else:
            prime = Card.prime_product_from_hand(cards)
            return self.table.unsuited_lookup[prime], cards

    def _six(self, cards):
        """
        Performs five_card_eval() on all (6 choose 5) = 6 subsets
        of 5 cards in the set of 6 to determine the best ranking, 
        and returns this ranking.
        """
        minimum = LookupTable.MAX_HIGH_CARD
        min_i = -1
        
        all5cardcombobs = list(itertools.combinations(cards, 5))
        for i, combo in enumerate(all5cardcombobs):

            score = self._five(combo)[0]
            if score <= minimum:
                minimum = score
                min_i = i

        return minimum, all5cardcombobs[i]


    def _seven(self, cards):
        """
        Performs five_card_eval() on all (7 choose 5) = 21 subsets
        of 5 cards in the set of 7 to determine the best ranking, 
        and returns this ranking.
        """
        minimum = LookupTable.MAX_HIGH_CARD
        min_i = -1
        
        all5cardcombobs = list(itertools.combinations(cards, 5))
        for i, combo in enumerate(all5cardcombobs):
            
            score = self._five(combo)[0]
            if score < minimum:
                minimum = score
                min_i = i
        return minimum, all5cardcombobs[i]
