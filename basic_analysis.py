import pandas as pd
import numpy as np
from scripts.utils import Collection, Sentence
from pathlib import Path
from transformers import BatchEncoding, BertTokenizerFast, BertConfig
from tokenizers import Encoding
from mytools.DataRelation import DataRelation   
from mytools.Data import Data

pretrained_model_name = 'bert-base-multilingual-cased'
max_len=20

finput= Path('data/example')
collection = (
    Collection().load_dir(finput)
    if finput.is_dir()
    else Collection().load(finput)
)

ner = Data(max_len=max_len, max_len_char=2, char_dict='', pretrained_model_name=pretrained_model_name, char=False)
re = DataRelation(max_len)

_, tokens, token_span = ner.process_data(collection)
#print(tokens)
pairs, tags = re.process_input(collection, tokens, token_span)
print(np.array(pairs).shape)
print(np.array(tags).shape)
#get all different characters
"""
chars = set([char for sentence in collection.sentences for word in sentence.text for char in word])
print(chars)
print(''.join(chars))
"""

"""

#get the maximum length of a sentence (number of tokens)
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name,strip_accents=True)


maxi = 0
max_len_char = 0
sentence_token = None
chars_set = set()
for sentence in collection.sentences:
    tokenized_batch : BatchEncoding = tokenizer(sentence.text)
    tokenized_text :Encoding  =tokenized_batch[0]
    tokenss = tokenized_text.tokens
    
    len_char = max([len(token) for token in tokens])
    if max_len_char < len_char: 
        max_len_char = len_char
        sentence_token = tokenss
    
    chars = set([char for token in tokenss for char in token])
    chars_set.update(chars)
    
    length = len(tokenized_text.tokens)
    if maxi < length: maxi = length
        
#chars= set(chars)
print(chars_set)  
print(len(chars_set))
print(''.join(list(chars_set)))   

print(maxi)
print(max_len_char)
print(sentence_token)
"""
#'bert-base-multilingual-uncased'
"""
{'t', 'x', 'p', 'v', 'r', 'u', 'z', '-', '2', '5', 'S', ',', 'b', ':', ';', '.', '6', 'i', ')', '(', '/', ']', '3', 'l', 'w', 'h', 'c', 'm', 'd', 'g', '4', 'E', 'P', 'e', 'a', 'L', '1', '[', 
'o', 'n', 's', 'y', 'q', '9', 'C', '#', '"', 'k', 'f', '0', 'j', '7'}
52
txpvruz-25S,b:;.6i)(/]3lwhcmdg4EPeaL1[onsyq9C#"kf0j7
85
15
['[CLS]', 'el', 'bronce', '##ado', 'en', 'interior', '##es', 'es', 'particularmente', 'peligro', '##so', 'para', 'los', 'mas', 'jovenes', '.', '[SEP]']
"""

#'bert-base-multilingual-cased'
"""
{'w', 'e', 'E', 'U', 'j', 'H', 's', 't', 'k', 'b', '5', ']', 'f', 'g', 'V', 'C', ')', 'L', '#', 'z', 'T', 'u', 'c', 'X', '.', '0', 'B', 'G', '9', '4', 'i', 'n', 'O', '7', '"', 'D', '-', 'N', 
';', ',', '(', '[', '3', 'Y', 'x', 'M', 'd', 'I', 'q', 'S', 'l', '6', '1', '2', 'o', 'Z', 'a', 'r', 'v', '/', 'R', 'A', 'F', ':', 'm', 'h', 'y', 'P', 'p', 'K'}
70
weEUjHstkb5]fgVC)L#zTucX.0BG94inO7"D-N;,([3YxMdIqSl612oZarv/RAF:mhyPpK
92
15
['[CLS]', 'El', 'bronce', '##ado', 'en', 'interior', '##es', 'es', 'particularmente', 'peligro', '##so', 'para', 'los', 'mas', 'joven', '##es', '.', '[SEP]']


print(collection.sentences[0])
print(collection.sentences[0].relations)
for r in collection.sentences[0].relations:
    print(r.origin)
    print(r.destination)
    print('...')
"""