import pandas as pd
import numpy as np
import os
from collections import Counter
import logging, sys
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# load the file_path and return a pandas dataframe
def load_single_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def load_folder(dir_path):
    csv_dir = os.path.join(dir_path, 'csv')


def delta_to_char(delta, scale=0.01):
    if delta >= 0:
        # increase
        bracket = min(int(delta / scale), 25)
        c = chr(ord('A') + bracket)
    else:
        # decrease
        delta = -delta
        bracket = min(int(delta // scale), 25)
        c = chr(ord('a') + bracket)
    return c


def char_to_delta(c, scale=0.01):
    if c.isupper():
        # increase
        bracket = ord(c) - ord('A')
        delta = bracket*scale
    else:
        # decrease
        bracket = ord(c) - ord('a')
        delta = -bracket*scale
    return delta


# convert the time series to a string using % changes and the specified scale factor
def ts_to_string(ts, scale=0.05):
    deltas = [(b - a)/a for a, b in zip(ts[:-1], ts[1:]) if a != 0]

    # vectorized_d2c = np.vectorize(delta_to_char)
    # delta_chars = vectorized_d2c(deltas)

    delta_chars = ''.join([delta_to_char(delta) for delta in deltas])

    return delta_chars


# n=2 by default for bigrams
def ngram_counts(arr, n=2):
    ngrams = [tuple(arr[j] for j in range(i, i+n)) for i in range(len(arr)-(n-1))]
    ##
    # logging.info(f'{arr} wtih ngrams {ngrams}')
    ##
    return Counter(ngrams)


def BPE_flat(s_arr, n=100):
    tokenizer = {}
    vocab = list(set(s_arr))
    for i in range(n):
        corpus = ngram_counts(s_arr)
        pair, _ = corpus.most_common(1)[0]
    
        token = ''.join(pair)
        tokenizer[pair] = token
        vocab.append(token)
        
        j = 0
        while j < len(s_arr)-1:
            if (s_arr[j], s_arr[j+1]) == pair:
                # merge these two
                s_arr[j] = token
                s_arr.pop(j+1)
            j += 1
            
    return tokenizer


# takes as input an array of stock language (SL) strings
def BPE(input_arr, n=100, tokenizer=None):
    # split the input array into the individual bytes
    split_arr = [[char for char in SL_string] for SL_string in input_arr]

    # initialize the vocabulary
    vocab = set()
    for SL_arr in split_arr:
        vocab |= set(SL_arr)
    
    # initialize the tokenizer if one does not exist already
    if not tokenizer:
        tokenizer = {}
    
    for epoch in range(n):
        # collective count of all bigrams
        corpus = [ngram_counts(SL_arr) for SL_arr in split_arr]
        ## HACKY WAY TO COMBINE COUNTERS. CHANGE THIS IN THE FUTURE TO DNC METHOD
        corpus = sum(corpus, Counter())
        logging.info(f'Epoch {epoch+1} corpus:\n{corpus}')
        
        # find most common byte pair and record
        pair, _ = corpus.most_common(1)[0]
        token = ''.join(pair)
        tokenizer[pair] = token
        vocab.add(token)

        # merge the pair in our input
        for arr_num, SL_arr in enumerate(split_arr):
            removed = set()
            i = 0
            while i < len(SL_arr)-1:
                if ((SL_arr[i], SL_arr[i+1]) == pair):
                    # merge
                    SL_arr[i] = token
                    i += 1
                    removed.add(i)
                i += 1
            split_arr[arr_num] = [tok for i, tok in enumerate(SL_arr) if i not in removed]
    
    return tokenizer


def tokenize(s_arr, tokenizer):
    for pair, token in tokenizer.items():
        i = 0
        while i < len(s_arr)-1:
            if (s_arr[i], s_arr[i+1]) == pair:
                # merge these two
                s_arr[i] = token
                s_arr.pop(i+1)
            i += 1
    return s_arr


def plot_deltas(deltas):
    x = [i for i in range(len(deltas)+1)]
    y = [0]
    for delta in deltas:
        y.append(y[-1]+delta)
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x, y)
    ax.axis('equal')
    ax.axis('off')
    
    return fig, ax


def plot_delta_string(s):
    deltas = [char_to_delta(c) for c in s]
    fig, ax = plot_deltas(deltas)
    ax.set_title(f'{s}', fontsize=16)


def main():
    print('Stocks as a Language')


if __name__ == '__main__':
    main()