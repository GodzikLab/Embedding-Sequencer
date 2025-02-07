# Used to script actions embedding sequencer.

import modules.cluster_mapping
import modules.embedding_generation
import modules.file_io

import time

# inputs to consider
'''
- output type: fasta, aln, tsv (flags -f, -a, -t)
    - determine whether to call the directory function based on the number of inputs
- print flag: -p 
- path input > gets processed in the scripting (there's a function)
- path output (optional) - change the name of the output single file

main scripting function(input, output_type = f/a/t, print = True/False, output = "")
'''
def run_pipeline(input_path, output_type = "f", print = False, output = ""):
    '''Manages the scripting of actions for the embedding sequencer pipeline.'''
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

    # extract information from HDF file
    

    # build faiss index

    # establish/download model

    # LOOP - start output list

        # generate embeddings, normalize

        # perform ANN with N neighbors

        # determine cluster-label sequence

        # calculate outlier percentage

        # find repeat locations

        # save to output list

    # END LOOP

    # mirror output: single fasta, multiple fastas, csv/tsv files

    return