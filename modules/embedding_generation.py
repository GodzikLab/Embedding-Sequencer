# Functions used for generating and managing new embeddings 

import numpy as np
import re

import torch
from sklearn.preprocessing import normalize

# EMBEDDING GENERATION

def download_model(source = "facebookresearch/esm:main", version = "esm2_t12_35M_UR50D"):
    '''Downloads and sets up the model and batch_converter of the protein language model being used.'''
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D", verbose = False)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def generate_embeddings(sequence, model, batch_converter):
    '''Generates embeddings for a single sequence using prepared model and normalizes.'''
    batch_labels, batch_strs, batch_tokens = batch_converter([["", sequence]])
    num_layers = len(model.layers)

    with torch.no_grad():
        # pull last layer of embeddings
        results = model(batch_tokens, repr_layers=[num_layers])
        query_embeddings = results["representations"][num_layers][0, 1:-1].numpy()
        # normalize embeddings
        query_embeddings = normalize(query_embeddings)
        query_embeddings = query_embeddings.astype(np.float32)
    return query_embeddings

# PATTERN FINDER FUNCTIONS

def find_pattern(query_sequence, indicative_pattern):
    '''Finds the indices of the indicative pattern in the sequence. Returns the center indexes of the pattern.'''
    shift = (len(indicative_pattern)) // 2 # index shift for the pattern to be in the center position

    # regex search expression to find matches
    pattern_indexes = [match.start() + shift for match in re.finditer(f'(?={indicative_pattern})', query_sequence)]
    # checks for missed initial fragments if other patterns were found
    if len(pattern_indexes) >= 2:
        first_index = [match.start() for match in re.finditer(f'(?={indicative_pattern[shift-1:]})', query_sequence[:pattern_indexes[0]])]
        if first_index: pattern_indexes.insert(0, first_index[0])
        elif pattern_indexes[0] > 35:
            pattern_indexes.insert(0, pattern_indexes[0] - pattern_indexes[1] + pattern_indexes[0])
    else:
        pattern_indexes = [] # empties if less than 2 patterns were found / a false positive

    return pattern_indexes
