import json
import pickle
from collections import namedtuple, defaultdict
from typing import List, Tuple

import flask
from flask import request, Flask

from masked_word import predict_masked_sentence

app = Flask(__name__)


import torch
from transformers import AutoTokenizer, AutoModel, BertForNextSentencePrediction, BertForMaskedLM

def load_resource(remote_src : str, LoadingClass : type, local_src : str):
    try:
        resource = LoadingClass.from_pretrained(remote_src)
        with open(local_src, 'wb') as handle:
            pickle.dump(resource, handle)
    except Exception as e:
        print('Error fetching remote %s: %s' % (remote_src, str(e)))
        with open(local_src, 'rb') as handle:
            resource = pickle.load(handle)
    return resource

# tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1", )
# vector_model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# nsp_model = BertForNextSentencePrediction.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# mask_model = BertForMaskedLM.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# mask_model.eval()

remote_source = "dumitrescustefan/bert-base-romanian-cased-v1"
tokenizer = load_resource(remote_source, AutoTokenizer, './tokenizer.p')
vector_model = load_resource(remote_source, AutoModel, './vector_model.p')
nsp_model = load_resource(remote_source, BertForNextSentencePrediction, './nsp_model.p')
mask_model = load_resource(remote_source, BertForMaskedLM, './mask_model.p')

print('Done loading')

@app.route('/test', methods=['GET'])
def test():
    return """
    <html><body><p>This is a test</p></body></html>
    """

@app.route('/', methods=['POST'])
def serve_data():
    try:
        json_data = request.get_json()
    except Exception as e:
        return json.dumps({'status':'error', 'error':str(e)})
    ping = json_data.get('ping')
    if ping is not None:
        return json.dumps({'status':'ok', 'ping':str(ping)})
    try:
        words = json_data['text']
    except Exception as e:
        return json.dumps({'status': 'error', 'error': str(e) + ', no text found in input data'})
    try:
        is_split = not isinstance(words, str)
        tok_out = tokenizer(words, is_split_into_words=is_split, return_tensors='pt')
        if is_split:
            word_list = words
        else:
            word_list = []
            for token in tok_out.tokens():
                if token[0] == '[' and token[-1] == ']': continue
                if token.startswith('##'):
                    word_list[-1] += token[2:]
                else:
                    word_list.append(token)
        model_out = vector_model(tok_out['input_ids'])
        last_hidden_states = model_out.last_hidden_state[0].tolist()
        # assert len(last_hidden_states) == len(tok_out.word_ids())
        sentence_vector = last_hidden_states[0]
        num_words = max([i for i in tok_out.word_ids() if i is not None]) + 1
        word_vectors = [ [] for _ in range(num_words)  ] # list of empty lists
        for index, vector in zip(tok_out.word_ids(), last_hidden_states):
            if index is None:
                continue
            word_vectors[index].append(vector)
        # we only want the first vector
        word_vectors = [v[0] for v in word_vectors]
        return_string = json.dumps({'status':'ok', 'sentence_vector':sentence_vector,
                           'word_vectors': word_vectors, 'words':word_list})
    except Exception as e:
        return json.dumps({'status':'error', 'error':str(e)})
    return return_string

@app.route('/nsp', methods=['POST'])
def serve_nsp():
    try:
        json_data = request.get_json()
    except Exception as e:
        return json.dumps({'status':'error', 'error':str(e)})
    try:
        text1 = json_data['text1']
        text2 = json_data['text2']
    except Exception as e:
        return json.dumps({'status': 'error', 'error': '"%s"' % str(e) + ', text not found in input data'})

    try:
        toks = tokenizer(text1, text2, is_split_into_words=False, return_tensors='pt')
        model_out = nsp_model(**toks)
        prob = torch.softmax(model_out.logits, dim=1)[0][0].item()
    except Exception as e:
        return json.dumps({'status': 'error', 'error': 'Error calculating probability: "%s"' % str(e)})

    return {'status':'ok', 'probability':str(prob)}

@app.route('/mask', methods=['POST'])
def serve_mask():
    try:
        json_data = request.get_json()
    except Exception as e:
        return json.dumps({'status':'error', 'error':str(e)})
    try:
        text = json_data['text']
        num_results = json_data.get('num_results')
    except Exception as e:
        return json.dumps({'status': 'error', 'error': '"%s"' % str(e) + ', text not found in input data'})

    try:
        if not num_results: num_results = 5
        mask_guesses = predict_masked_sentence(text, tokenizer, mask_model, num_results)
    except Exception as e:
        return json.dumps({'status': 'error', 'error': '"%s"' % str(e)})

    return json.dumps({'status':'ok', 'results':mask_guesses})

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(port=5001) #, use_reloader=False)
