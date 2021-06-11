import numpy as np
from typing import List
from bisect import bisect
from scripts.utils import Collection, Keyphrase, Relation, Sentence
import logging
logger = logging.getLogger(__name__)


def encode_bilou(sentence: Sentence, token_spans): 
    """
    Devuelve las etiquetas (bilou y labels) acorde al texto tokenizado
    """
    sent_len = np.array(token_spans).shape[0] # token_spans=[ [(1,2),(3,6), (7,8), (9,11)], [(0,4),(5,9)], ..]  spans = [ [(1,6),(7,11)], ...]
    token_span_START = [k[0] for k in token_spans]

    aligned_labels = ["O"]*sent_len 
    aligned_bilou = ["O"]*sent_len 
 
    for keyphrase in sentence.keyphrases:
        spans = keyphrase.spans
        n = len(spans) # n>1 when keyphrases consists of more than one word
        if n==1: 
            bilou = ["U-"]
        else:
            bilou = ["B-"]
            bilou.extend(["I-"]*(n-2))
            bilou.extend(["L-"])
        
        for i, span in enumerate(spans): 
            #only the first token that consitutes the word is labelled (when level of detail si token)
            token_ix = bisect(token_span_START, span[0]) - 1 # a=[0,3,5] x=4 -> bisect(a,x)= 2                   
            aligned_bilou[token_ix] = bilou[i] 
            aligned_labels[token_ix] = keyphrase.label
    return aligned_bilou, aligned_labels                    

   
def decode_bilou(sentence:Sentence, tags, tokens, spans) -> List[Keyphrase]:
    """tags: B-Concept, B-Action, ..."""

    next_id = 0 # unique id 
    tokens = [{"token": i,"span":j,"label":k} for i,j,k in zip(tokens, spans, tags) if j!= (0,0)]
    
    entity_spans = []
    entity_label = None
    prev_state = None
    prev_label = 'O'

    # if tokens are the atomic elements, we convert them to word
    words = []
    for token in tokens:
        if token['token'].startswith('##'):
            #  as ##ma is being convert to asma
            word = words.pop()
            s0 = list(word['span'])[0]
            s1 = list(token['span'])[1]
            words.append({"token":word['token']+token['token'][2:],"span":(s0, s1),"label":word['label']})
        else:
            words.append(token)

    list_of_keypfrases = []
    for w in words: 
        """IF the new label is BUO, 
        or the new label is different from previous one, 
        or  the new label is IL but the previuos one isnt BI
        THEN save the previous entity and reset """
        bool_1 = (w['label'][:1] in ['B','U','O'])
        bool_2 = w['label'][2:] != prev_label
        bool_3 = (w['label'][:1] in ['I','L']) and (prev_state not in ['B','I'])
        if bool_1 or bool_2 or bool_3:
            if entity_spans:
                keyphrase = Keyphrase(sentence=sentence, label=entity_label, id = next_id, spans=entity_spans)
                list_of_keypfrases.append(keyphrase)     
                next_id += 1                
                entity_spans = []
        
        if w['label'] == 'O':
            entity_label = 'O'            
        else:    
            entity_spans.append( w['span'] )
            entity_label = w['label'][2:]                
            
        prev_state = w['label'][:1]
        prev_label = entity_label
    return list_of_keypfrases


    