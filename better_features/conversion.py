from deuces.card import Card

def pypoker_to_deuces_str(s):
    return s[1] + s[0].lower()

def pypoker_to_deuces_strlist(L):
    return [pypoker_to_deuces_str(x) for x in L]

def to_deuces_intlist(L):
    return [Card.new(z) for z in pypoker_to_deuces_strlist(L)]