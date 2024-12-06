from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.preprocessing import normalize

import faiss
import h5py
import torch

import numpy as np
import re

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process protein sequences and generate FASTA output.")
    parser.add_argument("-f", "--output_flag", choices=['f', 's'], default='f', help="Choose between 'f' (find) or 's' (sequence)")
    parser.add_argument("-p", "--print_flag", action="store_true", help="Enable printing of intermediate results")
    parser.add_argument("input_hdf", help="Path to the input HDF file")
    parser.add_argument("input_fasta", help="Path to the input FASTA file")
    parser.add_argument("output_fasta", help="Path to the output FASTA file")

    args = parser.parse_args()
    return args

# used to write fastas with newlines based on break_indexes
def write_fasta_files(gene_id, sequence, break_indexes, output_name = 'output.fasta'):
    # if indexes is blank, uses normal writing schema
    if not break_indexes:
        seq_record = SeqRecord(Seq(sequence), id=gene_id, description="")
        with open(output_name, "w") as fasta_file:
            SeqIO.write(seq_record, fasta_file, "fasta")

    # uses newlines at points of break indexes 
    else:
        with open(output_name, "w") as fasta_file:
            fasta_file.write(f">{id} \n")

            start = 0
            for idx in break_indexes:
                if start >= len(sequence):
                    break
                segment = sequence[start:idx]
                fasta_file.write(segment + "\n")
                # print(f"Wrote to {idx}")
                start = idx

            if start < len(sequence):
                fasta_file.write(sequence[start:])

def main():
    # parse command line arguments
    args = parse_arguments()

    # unpackage data from hdf file
    with h5py.File(args.input_hdf, 'r') as f:
        aggregate_embeddings = f['embeddings'][:]
        cluster_labels = f['labels'][:]
        pool_proteins_list = f['proteins_list'][:]
        indicative_pattern = f['pattern'][()].decode('utf-8')
        pattern_percentage = f['pattern_percentage'][()]
    pattern_percentage = round(pattern_percentage, 3)

    # check datatypes & dimensionality
    if not isinstance(aggregate_embeddings, np.ndarray) or aggregate_embeddings.dtype != np.float32 or aggregate_embeddings.ndim != 2:
        raise ValueError("Error with 'embeddings' data in HDF file. Check data type and dimensions.")
    if not isinstance(cluster_labels, np.ndarray) or cluster_labels.dtype != np.int32 or cluster_labels.ndim != 1:
        raise ValueError("Error with 'labels' data in HDF file. Check data type and dimensions.")
    if not isinstance(pool_proteins_list, np.ndarray) or pool_proteins_list.dtype != np.object_ or pool_proteins_list.ndim != 1:
        raise ValueError("Error with 'protein_data' data in HDF file. Check data type and dimensions.")
    if not isinstance(indicative_pattern, str):
        raise ValueError("Error with 'pattern' data in HDF file. Check data type.")
    if not isinstance(pattern_percentage, float):
        raise ValueError("Error with 'pattern_percentage' data in HDF file. Check data type.")

    # parse fasta file for gene and sequence
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        gene_name = record.id
        sequence = str(record.seq)
    if args.print_flag:
        print(f"Gene Name: {gene_name}")
        print(f"Sequence: {sequence}")

    # generate embeddings with torch
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([[gene_name, sequence]])
    num_layers = len(model.layers)

    # pull last layer of embeddings and normalize
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers])
        query_embeddings = results["representations"][num_layers][0, 1:].numpy()
        query_embeddings = normalize(query_embeddings)
        query_embeddings = query_embeddings.astype(np.float32)
    if args.print_flag:
        print(f"Generated Embeddings for {gene_name}. Shape: {query_embeddings.shape}")

    # index pool embeddings in FAISS
    faiss_index = faiss.IndexFlatIP(aggregate_embeddings.shape[1])
    faiss_index.add(aggregate_embeddings)
    if args.print_flag:
        print(f"FAISS index created successfully. Number of vectors: {faiss_index.ntotal}")
    
    # perform approximate nearest neighbor matching with faiss
    num_neighbors = 50
    faiss_similarity, faiss_indices = faiss_index.search(query_embeddings, num_neighbors)

    # build cluster label sequence based on faiss search
    query_sequence = ""
    outlier_dict = {}
    for i in range(len(faiss_similarity)):
        label_weights = {}
        for sim, idx in zip(faiss_similarity[i], faiss_indices[i]):
            label = cluster_labels[idx]
            label_weights[label] = label_weights.get(label,0) + sim # adds weight based on similarity score
        majority_label = max(label_weights, key = label_weights.get)
        majority_weight = label_weights[majority_label]
        query_sequence += chr(ord('A') + majority_label)
        if majority_weight < (num_neighbors * 0.30): # 0.3 is definition of 'bad similarity' for normalized vector comparisons
            outlier_dict[i] = majority_weight
    
    # get outlier percentage
    outlier_percentage = round((len(outlier_dict) / len(faiss_similarity)) * 100, 3)
    sequence_confidence = round(100 - outlier_percentage, 3)
    if args.print_flag:
        print(f"Sequence Confidence: {sequence_confidence}")

    # BREAK BASED ON FLAG
    if args.output_flag == 's':
        write_fasta_files(gene_name, query_sequence, [], output_name = args.output_fasta)
        if args.print_flag:
            print(f"Sequence FASTA generated: {args.output_fasta}")
        exit
    
    # search through generated query sequence
    if args.print_flag:
        print(f"Pattern: {indicative_pattern}")
        print(f"Pattern Confidence: {pattern_percentage}")
    shift = (len(indicative_pattern)) // 2
    # regex search expression to find matches
    pattern_indexes = [match.start() + shift for match in re.finditer(f'(?={indicative_pattern})', query_sequence)]
    if len(pattern_indexes) >= 2: # checks for missed initial fragments if other patterns were found
        first_index = [match.start() for match in re.finditer(f'(?={indicative_pattern[shift:]})', query_sequence[:pattern_indexes[0]])]
        if first_index: pattern_indexes.insert(0, first_index[0])
    elif args.print_flag: # if no matches were found
        print("No pattern found")
    
    write_fasta_files(gene_name, query_sequence, pattern_indexes, args.output_fasta)
    if args.print_flag:
        print(f"Locations of Interest Found. Output FASTA generated: {args.output_fasta}")

    return

if __name__ == "__main__":
    main()