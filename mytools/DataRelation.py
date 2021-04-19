import numpy as np
#from pathlib import Path
from itertools import product

import tensorflow as tf

from scripts.utils import Collection, Keyphrase, Relation, Sentence, ENTITIES, RELATIONS

class DataRelation:
    def __init__(self, max_len):
        self.max_len = max_len    
        self.rel = RELATIONS + ['none']
        self.n_rel = len(self.rel)
        self.rel2idx= {t: i for i, t in enumerate(self.rel)}
        self.idx2rel = {i: w for w, i in self.rel2idx.items()}

    def get_n_rel(self):
        return self.n_rel

    def process_output(self, collection: Collection, prediction,  token_pairs):    
        #token_pairs tienen el siguiente formato
        #[[ [{'text': asma, 'span': (4,8)},{'text': es, 'span': (9,11)}],...],...]
        predicted_tags = [[self.idx2rel.get(p) for p in pr] for pr in prediction]
        collection = decode_relations(collection, token_pairs, predicted_tags)
        return collection

    def process_input(self, collection: Collection, tokens, token_spans):
        sentence_relations = [] #n_sentences x max_len^2
        token_pairs = []
        for i, (tokens_s, token_spans_s) in enumerate(zip(tokens, token_spans)):
            sentence = collection.sentences[i] 
            T = [{"text":t, "span":s} for t, s in zip(tokens_s, token_spans_s)] #n_sentences x max_len   
            rel = [self.rel2idx['none']]*(self.max_len**2)
            k=0
            pairs=[]
            for t1 in T:
                for t2 in T:                    
                    if t1['text'] != t2['text'] and not t1['text'].startswith('##') \
                        and not t2['text'].startswith('##') \
                        and t1['text'] not in ['[PAD]','[SEP]','[CLS]'] \
                        and t2['text'] not in ['[PAD]','[SEP]','[CLS]']:
                        span1 = t1["span"]
                        span2 = t2["span"]
                        id_origin = findKeyphraseId(sentence, span1)
                        id_dest = findKeyphraseId(sentence, span2)
                        #puede que el token no sea keyphrase
                        if id_origin and id_dest and id_origin != id_dest:
                            relation = findRelation(sentence, id_origin, id_dest)
                            if relation:
                                rel[k] = self.rel2idx[relation.label]
                    k +=1
                    pairs.extend([(t1,t2)])
            sentence_relations.append(np.array(rel).astype('int64'))
            token_pairs.append(pairs)
            
        return token_pairs, sentence_relations

#puede haber 2 relaciones distintas entre dos Keuphrase fijos??
#voy a suponer que solo existe una unica relacion A->B
def findRelation(sentence: Sentence, orig, dest):
    for r in sentence.relations:
        if r.origin == orig and r.destination == dest:
            return r

#rel_tags es lista de str
def decode_relations(collection, token_pairs, rel_tags):
    for i, pair, tag in enumerate(zip(token_pairs, rel_tags)):
        sentence = collection.sentences[i]
        token1 = pair[0]['text']
        span1 = pair[0]['span']
        token2 = pair[1]['text']
        span2 = pair[1]['span']
        if tag != 'none' and token1 != token2 and not token1.startswith('##') and not token2.startswith('##'):
            id_kph1 = findKeyphraseId(sentence, span1)
            id_kph2 = findKeyphraseId(sentence, span2)
            #ppuede ocurrir que Keyphrase no exista para un token dado,
            #si el modelo no lo ha clasificado, por lo que id = None
            #en caso de entidades compuestas, puede ocurrir id_kph1 = id_kph2, descartamos esa relacion
            if id_kph1 and id_kph2 and id_kph1 != id_kph2:
                relation = Relation(sentence=sentence, origin = id_kph1, destination = id_kph2, label = tag)
                sentence.relations.append(relation)
    return collection


def findKeyphraseId(sentence: Sentence, span: tuple):
    #devuleve None si no encuentra Keyphrase
    for kph in sentence.keyphrases:
        #(1,3) pertence a (1,5);(9,12)
        #(9,12) no pertence a (3,5);(9,12), ya que marcamos las relaciones entre el primer token de la frase
        #(5,10) no pertence a (3,5);(9,12)        
        s = kph.spans[0]
        if span[0] >= s[0] and span[1] <= s[1]:
            return kph.id

def all_vs_all_pairs(a):
    a = tf.convert_to_tensor(a) #n_sent x max_len x embedd
    tile_a = tf.tile(tf.expand_dims(a, 2), [1,1,a.get_shape()[1],1]) 
    tile_a = tf.expand_dims(tile_a, 3) 
    tile_b = tf.tile(tf.expand_dims(a, 1), [1,a.get_shape()[1],1,1]) 
    tile_b = tf.expand_dims(tile_b, 3) 
    prod = tf.concat([tile_a, tile_b], axis=3) 
    out = tf.reshape(prod,(-1,a.get_shape()[1]*a.get_shape()[1],2*a.get_shape()[2]))
    return out