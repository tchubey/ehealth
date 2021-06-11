import numpy as np
from typing import List
#from itertools import product
#import tensorflow as tf

from scripts.utils import Collection, Relation, Sentence, RELATIONS

class DataRelation:
    def __init__(self, max_len):
        self.max_len = max_len    
        pad = {"pad": 0} 
        self.rel = ['none'] + RELATIONS
        self.rel2idx= {**{t: i+1 for i, t in enumerate(self.rel)}, **pad}
        self.idx2rel = {i: w for w, i in self.rel2idx.items()}
        self.n_rel = len(self.rel2idx)


    def get_n_rel(self):
        return self.n_rel
   
    def process_input_sentence_4d(self, sentence: Sentence, tokens, spans) -> dict:
        T = [{"text":t, "span":s} for t, s in zip(tokens, spans)]    
        labels = []
        pairs=[]
        for t1 in T:
            rel = [self.rel2idx['pad']]*self.max_len
            for k, t2 in enumerate(T): 
                if t1['text'] not in ['[PAD]','[SEP]','[CLS]'] \
                        and t2['text'] not in ['[PAD]','[SEP]','[CLS]'] \
                        and t1['text'] != t2['text']  \
                        and not t1['text'].startswith('##') \
                        and not t2['text'].startswith('##'):
                    span1 = t1["span"]
                    span2 = t2["span"]
                    id_origin = findKeyphraseId(sentence, span1)
                    id_dest = findKeyphraseId(sentence, span2)
                    #puede que el token no sea keyphrase
                    if id_origin and id_dest and id_origin != id_dest:
                        relation = findRelation(sentence, id_origin, id_dest)
                        if relation:
                            rel[k] = self.rel2idx[relation.label]
                        else:
                            rel[k] = self.rel2idx["none"]
                pairs.extend([(t1,t2)])
            labels.append(rel)
        return {"token_pairs": pairs, \
                "labels": np.array(labels) if labels else None
                }

    def process_input_collection_4d(self, collection: Collection, tokens, token_spans):
        relations = [] #n_sentences x max_len^2
        token_pairs = []
        for i, (tokens_s, spans_s) in enumerate(zip(tokens, token_spans)):
            sentence = collection.sentences[i] 
            out_dict = self.process_input_sentence_4d(sentence, tokens_s, spans_s)           
            relations.append(out_dict.get("labels"))
            token_pairs.append(out_dict.get("token_pairs"))
            
        return {"labels": np.array(relations).astype('int32') if relations else None,
                "token_pairs":token_pairs
                }

    def process_output_sentence(self, sentence, prediction, token_pairs) -> List[Relation]:
        predicted_tags = [self.idx2rel.get(p) for pred in prediction for p in pred]   
        #print(predicted_tags)
        list_of_relations = []
        for pair, tag in zip(token_pairs, predicted_tags):
            token1 = pair[0].get('text')
            span1 = pair[0]['span']
            token2 = pair[1].get('text')
            span2 = pair[1]['span']

            if tag != 'none' and token1 != token2 \
                    and not token1.startswith('##') and not token2.startswith('##') \
                    and token1 not in ['[PAD]','[SEP]','[CLS]'] and token2 not in ['[PAD]','[SEP]','[CLS]'] :
                id_kph1 = findKeyphraseId(sentence, span1)
                id_kph2 = findKeyphraseId(sentence, span2)
                # If NER module didnt identify an entity,  ID = None
                # Also, if the pair of tokens belong to the same entity, they have same ID -> we remove this case 
                if id_kph1 and id_kph2 and id_kph1 != id_kph2:
                    relation = Relation(sentence=sentence, origin = id_kph1, destination = id_kph2, label = tag)
                    list_of_relations.append(relation)
        return list_of_relations

#--------------------------------------------------------------------------#

def findKeyphraseId(sentence: Sentence, span: tuple) -> int or None:
    for kph in sentence.keyphrases:
        # we consider only the first token of entity to establish the relationship       
        s = kph.spans[0]
        if span[0] >= s[0] and span[1] <= s[1]:
            return kph.id

# here we assume that there is only one possible directed relation between two entities
def findRelation(sentence: Sentence, orig, dest) -> Relation:
    for r in sentence.relations:
        if r.origin == orig and r.destination == dest:
            return r
