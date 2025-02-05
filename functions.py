import numpy as np

from sklearn.preprocessing import normalize # used to normalize embeddings

import h5py # working with HDF files
import faiss # index used for approx NN search of vectors
import torch # for managing models

# CLUSTERING POOL / FAISS INDEX FUNCTIONS

def unpack_hdf(hdf_file):
    '''Extracts the information of an inputted HDF file and checks their correctness.
    
    Embeddings, clustering labels, proteins in the pool, pattern, percentage occurence of pattern
    '''
    with h5py.File(hdf_file, 'r') as f:
        aggregated_embeddings = f['embeddings'][:]
        cluster_labels = f['labels'][:]
        pool_proteins_list = f['proteins_list'][:]
        indicative_pattern = f['pattern'][()].decode('utf-8')
        pattern_percentage = f['pattern_percentage'][()]
    pattern_percentage = round(pattern_percentage, 3)

    if not isinstance(aggregated_embeddings, np.ndarray) or aggregated_embeddings.dtype != np.float32 or aggregated_embeddings.ndim != 2:
        raise ValueError("Error with 'embeddings' data in HDF file. Check data type and dimensions.")
    if not isinstance(cluster_labels, np.ndarray) or cluster_labels.dtype != np.int32 or cluster_labels.ndim != 1:
        raise ValueError("Error with 'labels' data in HDF file. Check data type and dimensions.")
    if not isinstance(pool_proteins_list, np.ndarray) or pool_proteins_list.dtype != np.object_ or pool_proteins_list.ndim != 1:
        raise ValueError("Error with 'protein_data' data in HDF file. Check data type and dimensions.")
    if not isinstance(indicative_pattern, str):
        raise ValueError("Error with 'pattern' data in HDF file. Check data type.")
    if not isinstance(pattern_percentage, float):
        raise ValueError("Error with 'pattern_percentage' data in HDF file. Check data type.")

    return aggregated_embeddings, cluster_labels, pool_proteins_list, indicative_pattern, pattern_percentage

def build_faiss_index(aggregated_embeddings):
    ''' Takes an input of aggregated embeddings to build a FAISS index for rapid approximate nearest neighbor search.'''
    faiss_index = faiss.IndexFlatIP(aggregated_embeddings.shape[1])
    faiss_index.add(aggregated_embeddings)
    return faiss_index

def search_faiss_index(query_embeddings, faiss_index, num_neighbors = 50):
    '''Searches prebuilt FAISS index with approximate nearest neighbor.'''
    faiss_similarity, faiss_indices = faiss_index.search(query_embeddings, num_neighbors)
    return faiss_similarity, faiss_indices

def build_faiss_sequence(faiss_similarity, faiss_indices, cluster_labels, num_neighbors = 50, bad_similarity_standard = 0.3):
    '''Builds the cluster-label sequence for the query embedding and returning additional outlier information.'''
    query_sequence = ""
    outlier_dict = {}
    # iterates through similar vectors
    for i in range(len(faiss_similarity)):
        label_weights = {}
        for sim, idx in zip(faiss_similarity[i], faiss_indices[i]):
            label = cluster_labels[idx]
            label_weights[label] = label_weights.get(label,0) + sim # adds voting weight based on similarity score
        majority_label = max(label_weights, key = label_weights.get)
        majority_weight = label_weights[majority_label]
        query_sequence += chr(ord('A') + majority_label)
        if majority_weight < (num_neighbors * bad_similarity_standard): # 0.3 is definition of 'bad similarity' for normalized vector comparisons
            outlier_dict[i] = majority_weight
    
    # get sequence confidence (percentage of sequence without outliers)
    outlier_percentage = round((len(outlier_dict) / len(faiss_similarity)) * 100, 3)
    sequence_confidence = round(100 - outlier_percentage, 3)
    return query_sequence, sequence_confidence


# NEW EMBEDDINGS FUNCTIONS

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