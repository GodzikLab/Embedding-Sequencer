# Functions used for generating and managing new embeddings 

import numpy as np
import re

import torch
from sklearn.preprocessing import normalize

from transformers import T5Tokenizer, T5EncoderModel

# MODEL MANAGEMENT

def download_model(version = "esm2_t12_35M_UR50D"):
    '''Directs the download and set-up of the correct protein language model.'''
    if version.startswith("esm"):
        model, converter = download_esm_model(version)
        return model, converter
    else:
        model, tokenizer = download_prot_model(version)
        return model, tokenizer

def generate_embeddings(sequence, model, converter_or_tokenizer, version):
    '''Generates embeddings for a single sequence using the prepared model and normalizes.'''
    if version.startswith("esm"):
        return generate_esm_embeddings(sequence, model, converter_or_tokenizer)
    else:
        # print(f"Length of sequence: {len(sequence)}")
        sequence = add_sequence_spacing(sequence)
        return generate_prot_embeddings(sequence, model, converter_or_tokenizer)

# EMBEDDING GENERATION

def add_sequence_spacing(sequence):
    '''Adds spaces between each residue in a sequence for the Prot model.'''
    return " ".join(sequence)

def download_prot_model(version = "Rostlab/prot_t5_xl_uniref50"):
    '''Downloads and sets up the Prot model and tokenizer of the protein language model being used.'''
    tokenizer = T5Tokenizer.from_pretrained(version, do_lower_case = False, legacy = True)
    model = T5EncoderModel.from_pretrained(version)
    model.eval()
    return model, tokenizer

def generate_prot_embeddings(sequence, model, tokenizer):
    '''Generates embeddings for a single sequence using prepared model and normalizes.'''
    encodings = tokenizer(sequence, return_tensors="pt", truncation = False, padding = True)
    encodings = encodings.to(model.device)

    with torch.no_grad():
        outputs = model(**encodings)
        embeddings = outputs.last_hidden_state[:, 0:-1, :].squeeze(0) # removes end tokens
        embeddings = normalize(embeddings) 
        embeddings = embeddings.astype(np.float32)
    # print(embeddings.shape)
    return embeddings

def download_esm_model(version = "esm2_t12_35M_UR50D"):
    '''Downloads and sets up the ESM model and batch_converter of the protein language model being used.'''
    model, alphabet = torch.hub.load("facebookresearch/esm:main", version, verbose = False)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def generate_esm_embeddings(sequence, model, batch_converter):
    '''Generates embeddings for a single sequence using prepared model and normalizes.'''
    _, _, batch_tokens = batch_converter([["", sequence]])
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
        first_index = [match.start() for match in re.finditer(f'(?={indicative_pattern[shift:]})', query_sequence[:pattern_indexes[0]])]
        if first_index: pattern_indexes.insert(0, first_index[0])
        elif pattern_indexes[0] > 35:
            pattern_indexes.insert(0, pattern_indexes[0] - pattern_indexes[1] + pattern_indexes[0])
    else:
        pattern_indexes = [] # empties if less than 2 patterns were found / a false positive

    return pattern_indexes
