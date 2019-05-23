import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json

def get_embedding(sentence):
    outF = open("input.txt", "w")
    outF.write(sentence)
    outF.close()

    os.system('python biobert-master/extract_features.py \
    --input_file=input.txt \
    --vocab_file=pubmed_pmc_470k/vocab.txt \
    --bert_config_file=pubmed_pmc_470k/bert_config.json \
    --init_checkpoint=pubmed_pmc_470k/biobert_model.ckpt \
    --layers=-1 \
    --output_file=output.jsonl')

    with open('output.jsonl') as f:
        d = json.load(f)
    os.system('rm input.txt')
    os.system('rm output.jsonl')

    sentence_vector = np.zeros(768)
    for i in range (1, len(d['features'])-1):
        sentence_vector += sentence_vector + d['features'][i]['layers'][0]['values']
    num_tokens = len(d['features']) - 2
    vec = sentence_vector/num_tokens
    vec = vec.reshape((768,1))

    return vec

dayOne = pd.read_pickle("dayOneNotes_first.pkl")
notes = dayOne['NOTE']
out = [i for i in range(1500)]
n2 = notes[3500:5000] ##starting the 500th
c = 0
for no in n2:
    vec = get_embedding(no)
    out[c] = vec
    c += 1
    if c == 500:
        print(c)
        partial = pd.Series(out)
        parBert = pd.DataFrame([partial])
        parBert.to_pickle("parBert4k.pkl")
    if c == 1000:
        print(c)
        partial = pd.Series(out)
        parBert = pd.DataFrame([partial])
        parBert.to_pickle("parBert4-5k.pkl")
    if c == 1500:
        print(c)
        partial = pd.Series(out)
        parBert = pd.DataFrame([partial])
        parBert.to_pickle("parBert5k.pkl")
