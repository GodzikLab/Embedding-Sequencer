# Used to script actions embedding sequencer.

import modules.cluster_mapping
import modules.embedding_generation
import modules.file_io

import time
import pandas as pd
import os

def run_pipeline(input_path, hdf_file = "", output_type = "f", print_flag = False, output_path = "fasta_output"):
    '''Manages the scripting of actions for the embedding sequencer pipeline.
    
    Parameters:
        input_path: Input of the pipeline, takes FASTA, directory, or CSV/TSV
        hdf_file: Clustering mapping information as an HDF file
        output_type: Determines the output of the pipeline (f - FASTA, a - ALN, t - TSV file)
        print_flag: If True, prints progress updates within the pipeline.
        output_path: Determines the output folder
    '''
    if print_flag: print("Starting Embedding Sequencer Pipeline...\n")

    # timer
    start_time = round(time.perf_counter(), 4)

    # parse input, build input list, and determine input/output
    input_type = modules.file_io.determine_input(input_path) # 0 - FASTA, 1 - directory, 2 - CSV/TSV
    if input_type == 0:
        input_df = modules.file_io.read_fasta(input_path)
    elif input_type == 1:
        input_df = modules.file_io.read_directory_fastas(input_path)
    elif input_type == 2:
        input_df = modules.file_io.read_csv_tsv(input_path)
    else:
        raise ValueError("Input is not an acceptable format. Please enter a path for a FASTA, directory of FASTAs, or CSV/TSV.")
    
    # check input and output_type
    modules.file_io.validate_input_df(input_df) # checks df for columns and non-empty
    if print_flag: print(f"Successfully inputted query of {len(input_df)} proteins.\n")
    if print_flag: print(input_df + "\n")

    # extract clustering information from HDF file and build FAISS index
    aggregated_embeddings, saved_pca, cluster_labels, _, indicative_pattern, _, model_version = modules.cluster_mapping.unpack_hdf(hdf_file)
    if print_flag: print(f"Shape of Embeddings: {aggregated_embeddings.shape}")
    faiss_index = modules.cluster_mapping.build_faiss_index(aggregated_embeddings)
    if print_flag: print(f"Unpacked clustering information from {hdf_file} and built FAISS index.")

    # establish/download model
    model, converter_or_tokenizer = modules.embedding_generation.download_model(version = model_version)
    if print_flag: print("Finished downloading and setting up protein language model.\n")

    # start output list
    output_data = []
    num_neighbors = 50
    if print_flag: print("Generating Embeddings Sequences for Query Proteins...")

    # LOOP
    for _, row in input_df.iterrows():
        entry, name, sequence = row["Entry"], row["Entry Name"], row["Sequence"]

        # generate embeddings
        query_embeddings = modules.embedding_generation.generate_embeddings(sequence, model, converter_or_tokenizer, version = model_version)

        # apply PCA dimensionality reduction
        query_embeddings = modules.cluster_mapping.apply_pca(query_embeddings, saved_pca)
        # if print_flag: print(f"   {name} - {query_embeddings.shape}")

        # perform ANN with N neighbors
        faiss_similarity, faiss_indices = modules.cluster_mapping.search_faiss_index(query_embeddings, faiss_index, num_neighbors = num_neighbors)
        
        # determine cluster-label sequence
        query_sequence, outlier_dict = modules.cluster_mapping.build_faiss_sequence(faiss_similarity, faiss_indices, cluster_labels, num_neighbors = num_neighbors)

        # calculate outlier percentage
        # sequence_confidence = modules.cluster_mapping.calculate_confidence(outlier_dict, faiss_similarity)
        # if print_flag: print(f"   {name} - {len(outlier_dict)} outliers with {sequence_confidence}% confidence.")

        # find repeat locations
        pattern_indexes = modules.embedding_generation.find_pattern(query_sequence, indicative_pattern)
        if pattern_indexes: # check if pattern indexes if empty
            start_pos, end_pos = pattern_indexes[0] + 1, pattern_indexes[-1] + 24
        else:
            start_pos, end_pos = -1, -1

        # save to output list
        data_entry = {
            "Entry" : entry,
            "Entry Name" : name,
            "Sequence" : query_sequence,
            # "Sequence Confidence" : sequence_confidence,
            "Repeat Locations" : [x + 1 for x in pattern_indexes], # indexes to residue count
            "Number of Repeats" : len(pattern_indexes),
            "Start" : start_pos,
            "End" : end_pos
        }
        output_data.append(data_entry)
        if print_flag: print(f"   {name} - {query_sequence[:50]}")
    
    output_df = pd.DataFrame(output_data)
    if print_flag: print("Finished Generating Embeddings.\n")
    # if print_flag: print(output_df)

    # output
    if output_type == "f": # FASTA
        modules.file_io.write_fastas_to_directory(output_df, output_directory = f"{output_path}/")
    elif output_type == "a": # ALN
        modules.file_io.write_alns_to_directory(input_df, output_df, output_directory = f"{output_path}/")
    elif output_type == "t": # TSV
        modules.file_io.write_tsv(output_df, output_directory = f"{output_path}/")
    else:
        raise ValueError("Invalid output type entered.")
    
    end_time = round(time.perf_counter(), 4)
    runtime = round(end_time - start_time, 4)
    if print_flag: print(f"Program Completed. Exiting with Total Runtime of {runtime} seconds.")

    return 