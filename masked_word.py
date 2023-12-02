import json
import pickle
from collections import namedtuple, defaultdict
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel, BertForNextSentencePrediction, BertForMaskedLM, BertTokenizer

# load the tokenizer and the model


# print('Loading tokenizer')
# ro_tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
#
# print('Loading models')
# # model = AutoModel.from_pretrained("racai/distilbert-base-romanian-cased")
# # model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
#
# mask_model = BertForMaskedLM.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# mask_model.eval()

def predict_masked_sentence(text : str,
                        tokenizer : BertTokenizer,
                        model : BertForMaskedLM, top_k=5)\
        -> List[List[Tuple[str, float]]]:
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    masked_indices = [i for i, m in enumerate(tokenized_text) if m =='[MASK]']
    mask_replacement_list = []
    for masked_index in masked_indices: # tokenized_text.index("[MASK]")
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        replacement_options = []
        for i, pred_idx in enumerate(top_k_indices):
            predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
            token_weight = top_k_weights[i]
            replacement_options.append((predicted_token, float(token_weight)))
            # print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
        # print()
        mask_replacement_list.append(replacement_options)
    return mask_replacement_list