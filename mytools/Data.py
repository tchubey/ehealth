import numpy as np
#from pathlib import Path
from itertools import product

import tensorflow as tf
from tokenizers import Encoding
from transformers import BatchEncoding, BertTokenizerFast, BertConfig
from transformers import TFBertModel

from scripts.utils import Collection, Keyphrase, Relation, Sentence, ENTITIES
from mytools.process_bilou import encode_bilou, decode_bilou

class Data:
    def __init__(self, max_len, max_len_char, char_dict, pretrained_model_name, char=False):
        self.pretrained = pretrained_model_name
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained, strip_accents=True) # Load a pre-trained tokenizer
        self.bertembedd = TFBertModel.from_pretrained(self.pretrained)
        self.max_len = max_len
        self.max_len_char = max_len_char
        self.char_dict = char_dict
        self.char2idx = { **{"PAD":0,"UNK":1,"#":2}, **{c: i + 3 for i, c in enumerate(self.char_dict)}}
        self.tags = [''.join(x) for x in list(product(['B-','I-','L-','U-'], ENTITIES))] + ['O']
        self.tag2idx= {t: i for i, t in enumerate(self.tags)}
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}
        self.char = char

    def get_n_tags(self):
        return len(self.tag2idx)
     
    def get_n_chars(self):
        return len(self.char2idx)

    def get_embedd_dim(self):
        return 768
        
    def process_char(self, tokens):
        sent_seq = []
        for t in tokens:
            word_seq = []                
            for j in range(self.max_len_char):
                if t == "UNK": word_seq.append(self.char2idx.get("UNK"))
                elif t in ["[PAD]","[CLS]","[SEP]"]: word_seq.append(self.char2idx.get("PAD"))
                else:
                    try:
                        word_seq.append(self.char2idx.get(t[j]))
                    except:
                        word_seq.append(self.char2idx.get("UNK"))
            sent_seq.append(word_seq)
        return np.array(sent_seq)      
        
    def process_predictions(self, collection: Collection, prediction,  tokens, tokens_span):    
        predicted_tags = [[self.idx2tag.get(p) for p in pr] for pr in prediction]
        collection = decode_bilou(collection, tokens, tokens_span, predicted_tags)
        return collection
        
    def process_data(self, collection: Collection): 
        tokens = [] #token string: asma
        token_spans = [] #token span: (3,7)
        labels = [] #label: U-Concept
        attention_mask = [] 
        ids = []
        X_char = [] #[2,5,23,43,0]

        # FOR EACH SENTENCE DO :
        for sentence in collection.sentences:
            tokenized_batch : BatchEncoding = self.tokenizer(sentence.text, max_length=self.max_len , padding='max_length', truncation=True)
            tokenized_text :Encoding  =tokenized_batch[0]

            tokens.append(tokenized_text.tokens)
            token_spans.append(tokenized_text.offsets)

            labels.append(encode_bilou(sentence, tokenized_text.offsets))
            ids.append(tokenized_text.ids )
            attention_mask.append(tokenized_text.attention_mask)
            X_char.append(self.process_char(tokenized_text.tokens)) 
        #convertir labels into int values (dict)
        y_tags = [[self.tag2idx.get(l) for l in lab] for lab in labels] if labels else None
        y_tags = np.array(y_tags).astype('int64')
                
        embedding = self.bertembedd({'input_ids': np.array(ids),'attention_mask':  np.array(attention_mask)})
        X_word_embedd = np.array(embedding.last_hidden_state).astype('float64')
        #print(X_char)
        if self.char:
            X_char = np.array(X_char).astype('int64')
            return (X_word_embedd, X_char, y_tags), tokens, token_spans
        else:
            return (X_word_embedd, y_tags), tokens, token_spans

