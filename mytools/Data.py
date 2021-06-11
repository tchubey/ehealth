import numpy as np
from typing import List
from itertools import product
import string
import re
#import tensorflow as tf
from tokenizers import Encoding
from transformers import BatchEncoding, BertTokenizerFast
from transformers import TFBertModel

from scripts.utils import Collection, Keyphrase, Sentence, ENTITIES
from mytools.process_bilou import encode_bilou, decode_bilou

class Data:
    def __init__(self, max_len, pretrained_model_name, token2word):
        self.max_len = max_len
        self.pretrained = pretrained_model_name
        self.token2word = token2word

        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained, strip_accents=True) # Load a pre-trained tokenizer
        self.bertembedd = TFBertModel.from_pretrained(self.pretrained)

        pad = {"pad": 0} 
        self.labels =  ['O'] + ENTITIES
        self.label2idx= {**{t: i+1 for i, t in enumerate(self.labels)}, **pad}
        self.idx2label = {i: w for w, i in self.label2idx.items()}

        self.bilou = ['B-','I-','L-','U-', 'O']
        self.bilou2idx = {**{t: i+1 for i, t in enumerate(self.bilou)}, **pad}
        self.idx2bilou = {i: w for w, i in self.bilou2idx.items()}

    def get_n_labels(self):
        return len(self.label2idx)
     
    def get_n_bilou(self):
        return len(self.bilou2idx)

    def get_embedd_dim(self):
        return 768 #BERT falta automatizarlo 


    def process_input_sentence(self, sentence: Sentence) -> dict:
        text = sentence.text
       
        tokenized_batch : BatchEncoding = self.tokenizer(text)
        tokenized_text :Encoding  = tokenized_batch[0]
        ids = np.array([tokenized_text.ids])
        mask = np.array([tokenized_text.attention_mask]) # PAD = 0

        tokens = tokenized_text.tokens
        spans = tokenized_text.offsets
        embeddings = self.bertembedd({'input_ids': ids, \
                                    'attention_mask':  mask})
        if self.token2word:
            # take mean of token embeddings -> word embeddings
            embeddings, tokens, spans = process_token2word(embeddings, tokens, spans)
        else:
            embeddings = [e for e in embeddings.last_hidden_state.numpy()[0]]

        bilou, labels = encode_bilou(sentence, spans)

        #padding to max_len
        actual_len = len(tokens)
        if actual_len < self.max_len:
            n = self.max_len - actual_len
            word_pad = ['[PAD]']*n
            span_pad = [(0,0)]*n
            embedd_pad = np.zeros(shape=(n,self.get_embedd_dim()))
            tokens.extend(word_pad)
            spans.extend(span_pad)
            embeddings.extend(embedd_pad)
            bilou.extend(['pad']*n)
            labels.extend(['pad']*n)
        elif actual_len > self.max_len:
            tokens = tokens[:self.max_len]
            spans = spans[:self.max_len]
            embeddings = embeddings[:self.max_len]
            bilou = bilou[:self.max_len]
            labels = labels[:self.max_len]
                      
        #create mask. PAD = 1.0
        mask = [float(ii != '[PAD]') for ii in tokens]

        return {"tokens": np.array(tokens), \
                "spans": spans, \
                "labels": np.array([self.label2idx.get(l) for l in labels]) if labels else None, \
                "bilou": np.array([self.bilou2idx.get(l) for l in bilou]) if bilou else None, \
                "embeddings": np.array([embeddings]), \
                "mask": np.array([mask]),
                }

    def process_input_collection(self, collection: Collection) -> dict:
        tokens = [] #token string: asma (or as,##ma in case we dont perform embedding aggregation)
        spans = [] #token span: (3,7)
        labels = [] #Concept, Action, ...
        bilou = [] #BILOU tags
        embeddings = [] #vector of 768 length
        padding_mask = []
        
        for sentence in collection.sentences:
            processed_sentence = self.process_input_sentence(sentence)
            tokens.append(processed_sentence.get("tokens"))
            spans.append(processed_sentence.get("spans"))
            labels.append(processed_sentence.get("labels"))
            bilou.append(processed_sentence.get("bilou"))
            embeddings.extend(processed_sentence.get("embeddings"))
            padding_mask.extend(processed_sentence.get("mask"))

        return {"embeddings":np.array(embeddings).astype('float32'), \
                "mask": np.array(padding_mask), \
                "labels":np.array(labels).astype('int32'), \
                "bilou": np.array(bilou).astype('int32'), \
                "tokens":tokens, \
                "spans":spans
                }


    def process_output_sentence(self, sentence: Sentence, predicted_bilou, 
                                predicted_labels, tokens, spans) -> List[Keyphrase]:
        
        bilou = [self.idx2bilou.get(p) for p in predicted_bilou]
        #print(predicted_bilou)
        #print(bilou)
        labels = [self.idx2label.get(p) for p in predicted_labels]
               
        prediction = [b + t if b!='O' and t!='O' else 'O' for b, t in zip(bilou, labels)]

        return decode_bilou(sentence, prediction, tokens, spans)


def process_token2word(embeddings, tokens, spans):
    word_embeddings = []
    words = []
    word_span = []

    new_word = True
    txt = ""
    embedd = []
    span = [None, None]

    for t, s, e in zip(reversed(tokens), reversed(spans), \
                        reversed(embeddings.last_hidden_state.numpy()[0])):
        embedd.append(e)
        span[0] = s[0]
        if not span[1]: span[1] = s[1]

        if t.startswith('##'):
            txt = t[2:] + txt
        else:
            txt = t + txt
            new_word = True
        
        if new_word:
            word_embeddings.insert(0, np.mean(embedd, axis = 0))
            words.insert(0, txt)
            word_span.insert(0, tuple(span))
            new_word = False
            txt = ""
            embedd = []
            span = [None, None]

    return word_embeddings, words, word_span

    

  

