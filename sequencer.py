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
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process protein sequences and generate FASTA output.")
    parser.add_argument("-a", "--align_flag", action='store_true', help="Changes output from FASTA to CLUSTAL aln files.")
    parser.add_argument("-f", "--find_flag", action='store_const', const='f', default='s', help="Set to 'f' (find) if provided; defaults to 's' (sequence)")
    parser.add_argument("-p", "--print_flag", action="store_true", help="Enable printing of intermediate results")
    parser.add_argument("input_hdf", help="Path to the input HDF file")
    parser.add_argument("input_fasta", help="Path to the input FASTA file")
    parser.add_argument("output_fasta", help="Path to the output FASTA file")

    args = parser.parse_args()
    return args

# used to write fastas with newlines based on break_indexes
def write_fasta_files(gene_id, sequence, break_indexes, description = "", output_name = 'output.fasta'):
    # if indexes is blank, uses normal writing schema
    if not break_indexes:
        seq_record = SeqRecord(Seq(sequence), id = gene_id, description = description)
        with open(output_name, "w") as fasta_file:
            SeqIO.write(seq_record, fasta_file, "fasta")

    # uses newlines at points of break indexes 
    else:
        with open(output_name, "w") as fasta_file:
            fasta_file.write(f">{gene_id} \n")

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

def write_aln_files(seq1_id, seq2_id, seq1, seq2, break_indexes = [], output_name = 'output.aln'):
    # check sequence lengths are equal
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length.")

    # fix output name if fasta
    if output_name[-6:] == ".fasta":
        output_name = output_name[:-6] + ".aln"

    total_length = len(seq1)
    curr_pos = 0

    # if break_indexes not given, set list to increment in sets of 60
    if not break_indexes:
        increment = 60
        break_indexes = list(range(60, total_length + increment, increment))

    with open(output_name, "w") as file:
        # header
        file.write("CLUSTAL\n\n")

        # process each block
        for block_break in break_indexes:
            if curr_pos >= total_length:
                break # finish processing

            end_pos = min(block_break, total_length)

            block1 = seq1[curr_pos:end_pos]
            block2 = seq2[curr_pos:end_pos]

            file.write(f"{seq1_id.ljust(10)} {block1}\n")
            file.write(f"{seq2_id.ljust(10)} {block2}\n")
            file.write("\n")  # Add a blank line between blocks
            
            # Update current position
            curr_pos = end_pos

def main():
    prev_time = round(time.perf_counter(), 5)
    start_time = prev_time
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

    if args.print_flag:
        curr_time = round(time.perf_counter(), 5)
        print(f"HDF Unpacking Time: {curr_time - prev_time}")
        prev_time = curr_time

    # parse fasta file for gene and sequence
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        gene_name = record.id
        sequence = str(record.seq)
    if args.print_flag:
        print("\nEntered FASTA file:")
        print(f"Gene Name: {gene_name}")
        print(f"Sequence: {sequence}\n")

    if args.print_flag:
        curr_time = round(time.perf_counter(), 5)
        print(f"FASTA Unpacking Time: {curr_time - prev_time}")
        prev_time = curr_time

    # generate embeddings with torch
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D", verbose = False)
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([[gene_name, sequence]])
    num_layers = len(model.layers)

    # pull last layer of embeddings and normalize
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[num_layers])
        query_embeddings = results["representations"][num_layers][0, 1:-1].numpy()
        query_embeddings = normalize(query_embeddings)
        query_embeddings = query_embeddings.astype(np.float32)
    if args.print_flag:
        print(f"\nGenerated Embeddings for {gene_name}. Shape: {query_embeddings.shape}\n")

    if args.print_flag:
        curr_time = round(time.perf_counter(), 5)
        print(f"Embedding Generation Time: {curr_time - prev_time}\n")
        prev_time = curr_time

    # index pool embeddings in FAISS
    faiss_index = faiss.IndexFlatIP(aggregate_embeddings.shape[1])
    faiss_index.add(aggregate_embeddings)
    if args.print_flag:
        print(f"FAISS index created successfully. Number of Vectors in Pool: {faiss_index.ntotal}\n")
    
    # perform approximate nearest neighbor matching with faiss
    num_neighbors = 50
    faiss_similarity, faiss_indices = faiss_index.search(query_embeddings, num_neighbors)

    if args.print_flag:
        curr_time = round(time.perf_counter(), 5)
        print(f"\nFAISS Generation Time: {curr_time - prev_time}\n")
        prev_time = curr_time

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
        print(f"Sequence Confidence: {sequence_confidence}\n")

    if args.print_flag:
        curr_time = round(time.perf_counter(), 5)
        print(f"Cluster Label Sequencing Time: {curr_time - prev_time}\n")
        prev_time = curr_time

    # BREAK BASED ON FLAG
    if args.find_flag == 's':
        if args.align_flag:
            seq1_id = gene_name + "_AMINO"
            seq2_id = gene_name + "_EMBED"
            write_aln_files(seq1_id, seq2_id, sequence, query_sequence, [], output_name = args.output_fasta)
        else: 
            write_fasta_files(gene_name, query_sequence, [], output_name = args.output_fasta)
        if args.print_flag:
            print(f"Sequence FASTA generated: {args.output_fasta}")
        return
    
    # search through generated query sequence
    if args.print_flag:
        print(f"Pattern: {indicative_pattern}")
        print(f"Pattern Confidence: {pattern_percentage}\n")
    shift = (len(indicative_pattern)) // 2
    # regex search expression to find matches
    pattern_indexes = [match.start() + shift for match in re.finditer(f'(?={indicative_pattern})', query_sequence)]
    if len(pattern_indexes) >= 2: # checks for missed initial fragments if other patterns were found
        first_index = [match.start() for match in re.finditer(f'(?={indicative_pattern[shift-1:]})', query_sequence[:pattern_indexes[0]])]
        if first_index: pattern_indexes.insert(0, first_index[0])
        elif pattern_indexes[0] > 35:
            pattern_indexes.insert(0, pattern_indexes[0] - pattern_indexes[1] + pattern_indexes[0])
    elif args.print_flag: # if no matches were found
        print("No pattern found\n")
    
    description = ""
    if len(pattern_indexes) >= 2:
        description = f" | {len(pattern_indexes)} POI Found | Residues {pattern_indexes[0]+1} to {pattern_indexes[-1] + 24}"
    else:
        pattern_indexes = []

    if args.align_flag:
        seq1_id = gene_name + "_AMINO"
        seq2_id = gene_name + "_EMBED"
        write_aln_files(seq1_id, seq2_id, sequence, query_sequence, pattern_indexes, output_name = args.output_fasta)
    else: 
        write_fasta_files(gene_name + description, query_sequence, pattern_indexes, output_name = args.output_fasta)

    if args.print_flag:
        print(f"Output FASTA generated: {args.output_fasta}")
        print(f"\nLocations of Interest Indexes: {pattern_indexes}\n")

    if args.print_flag:
        curr_time = round(time.perf_counter(), 5)
        print(f"Finished Pattern Search: {curr_time - prev_time}\n")
        prev_time = curr_time

    if args.print_flag:
        curr_time = round(time.perf_counter(), 5)
        print(f"Total Time: {curr_time - start_time}\n")

    return

if __name__ == "__main__":
    main()