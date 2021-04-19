import numpy as np
from bisect import bisect
from scripts.utils import Collection, Keyphrase, Relation, Sentence
import logging
logger = logging.getLogger(__name__)


def encode_bilou(sentence: Sentence, token_spans): 
    inf = 1e5  
    max_len = np.array(token_spans).shape[0] # token_spans=[ [(1,2),(3,6), (7,8), (9,11)], [(0,4),(5,9)], ..]  spnas = [ [(1,6),(7,11)], ...]
    token_spans_0 = [k[0] if k[1] != 0 else inf for k in token_spans]

    aligned_labels = ["O"]*max_len # Make a list to store our labels the same length as max_len (as padding was done)

    for key in sentence.keyphrases:
        spans = key.spans
        n = len(spans)
        if n==1: 
            bilou = ["U-"]
        else:
            bilou = ["B-"]
            bilou.extend(["I-"]*(n-2))
            bilou.extend(["L-"])
            
        i = 0    
        for span in spans: #word, not phrase  
            annotation_token_ix_set = ( set() )
            for char_ix in range(span[0],span[1]):
                 
                token_ix = bisect(token_spans_0, char_ix) -1  
                
                if token_ix is not None: # White spaces have no token and will return None
                    annotation_token_ix_set.add(token_ix)

            bilou_i = bilou[i]
            i += 1
            
            if annotation_token_ix_set:
                token_ix = annotation_token_ix_set.pop()
                aligned_labels[token_ix] = bilou_i + key.label #only the first token that consitutes the word/phrase is labelled

    return aligned_labels                    

def decode_bilou(collection:Collection, tokens, spans, tags) -> Collection:
    next_id = 0
    #i = each sentence
    for i, (tok, sp, tag) in enumerate(zip(tokens, spans, tags)):
        sentence = collection.sentences[i]
        tokens = [{"token": i,"span":j,"label":k} for i,j,k in zip(tok, sp, tag) if j!= (0,0)]
        
        entity_spans = []
        entity_label = None
        prev_state = None
        prev_label = 'O'
        
        #juntamos los tokens en palabras, ignoramos las etiquetas de los tokens ## y solo nos quedamos con la etiqueta del primer token
        words = []
        for token in tokens:
            if token['token'].startswith('##'):
                #aqui suponemos que bert tokenizer funciona bien y ##token siempre va precedido por el token raiz
                word = words.pop()
                s0 = list(word['span'])[0]
                s1 = list(token['span'])[1]
                words.append({"token":word['token']+token['token'][2:],"span":(s0, s1),"label":word['label']})
            else:
                words.append(token)
        logger.debug(sentence)
        logger.debug(words)
        logger.debug('----------------------------------')
        #juntamos las frases segun el esquema BILOU (entidades continuas)
        for w in words:    
        
            bool_1 = (w['label'][:1] in ['B','U','O'])
            bool_2 = w['label'][2:] != prev_label
            bool_3 = (w['label'][:1] in ['I','L']) and (prev_state not in ['B','I'])
            if bool_1 or bool_2 or bool_3:
                #save previous entity and reset entity_spans and entity_label
                # no guardar vacio
                if entity_spans:
                    keyphrase = Keyphrase(sentence=sentence, label=entity_label, id = next_id, spans=entity_spans)
                    sentence.keyphrases.append(keyphrase)     
                    next_id += 1                
                    entity_spans = []
            
            if w['label'] == 'O':
                entity_label = 'O'
                
            else:    
                entity_spans.append( w['span'] )
                entity_label = w['label'][2:]                
                
            prev_state = w['label'][:1]
            prev_label = entity_label
    return collection
    



    