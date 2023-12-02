import json
import pickle
from collections import namedtuple, defaultdict
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel, BertForNextSentencePrediction

# load the tokenizer and the model


print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

print('Loading model')
# model = AutoModel.from_pretrained("racai/distilbert-base-romanian-cased")
nsp_model = BertForNextSentencePrediction.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# with open('./distilbert_model2.p', 'wb') as handle:
#     pickle.dump(model, handle)
# with open('./distilbert_model2.p', 'rb') as handle:
#     model = pickle.load(handle)
print('Done loading')

sentence1 = 'Ion e frumos.'
sentence2 = 'Maria e urâtă.'
sentence3 = 'Planeta se învârte neregulat în jurul supernovei.'

tok_out1 = tokenizer(sentence1, sentence2, is_split_into_words=False, return_tensors='pt')
tok_out2 = tokenizer(sentence1, sentence3, is_split_into_words=False, return_tensors='pt')

out1 = nsp_model(**tok_out1)
out2 = nsp_model(**tok_out2)
