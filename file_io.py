import os
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# INPUT FUNCTIONS

def determine_input(input_path):
    '''Determines the input type.
    0 - FASTA file
    1 - Directory (assumed multiple FASTA files)
    2 - CSV/TSV file
    '''
    if os.path.isdir(input_path):
        return 1 # path is directory
    if os.path.isfile(input_path):
        _, ext = os.path.splitext(input_path.lower())
        if ext in [".fa", ".fasta"]:
            return 0 # single FASTA file
        elif ext in [".csv", ".tsv"]:
            return 2 # CSV/TSV file
    return -1

def read_fasta(fasta_file):
    '''Extracts name and sequence information from FASTA file, building a DataFrame.'''
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        name = record.id
        sequence = str(record.seq)
        data.append({"Entry Name":name, "Sequence":sequence})
    df = pd.DataFrame(data)
    return df

def read_directory_fastas(directory):
    '''Reads every FASTA file's information in a directory, building a DataFrame with all items.'''
    data = []
    for file in os.listdir(directory):
        if file.endswith((".fasta", ".fa")):
            file_path = os.path.join(directory, file)
            try:
                for record in SeqIO.parse(file_path, "fasta"):
                    name = record.id
                    sequence = str(record.seq)
                    data.append({"Entry Name":name, "Sequence":sequence})
            except Exception as e:
                # print(f"Error processing {file}:{e}")
                continue # skips
    df = pd.DataFrame(data)
    return df

def read_csv_tsv(file_path):
    '''Takes in CSV or TSV files and then builds a DataFrame with the information.'''
    delimiter = "," # defaults to comma for CSV files
    if file_path.endswith(".tsv"): # switches to tab for TSV files
        delimiter = "\t"

    df = pd.read_csv(file_path, sep = delimiter, header = 0, usecols = [0, 1]) # header = 0 skips first row, assumes cols 0 and 1 have info
    df.columns = ["Entry Name", "Sequence"] # renames columns
    return df
    
# OUTPUT FUNCTIONS

def write_fasta(name, sequence, break_indexes = [], description = "", output_name = "output.fasta"):
    '''Writes a single FASTA file based on input of name and sequence. Uses special line breaks to show repeats.'''
    if not break_indexes: # normal fasta schema
        seq_record = SeqRecord(Seq(sequence), id = name, description = description)
        with open(output_name, "w") as fasta_file:
            SeqIO.write(seq_record, fasta_file, "fasta")
    else: # special line breaks
        with open(output_name, "w") as fasta_file:
            fasta_file.write(f">{name} \n")
            start = 0
            for idx in break_indexes:
                if start >= len(sequence):
                    break
                segment = sequence[start:idx]
                fasta_file.write(segment + "\n")
                start = idx
            if start < len(sequence):
                fasta_file.write(sequence[start:])
    return 0

def write_aln(name1, name2, seq1, seq2, break_indexes = [], output_name = "output.aln"):
    '''Writes a single ALN file based on input of 2 names and sequences.'''
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

        # add end block
        break_indexes.append(total_length)

        # process each block
        for block_break in break_indexes:
            if curr_pos >= total_length:
                break # finished processing
            end_pos = min(block_break, total_length)
            block1 = seq1[curr_pos:end_pos]
            block2 = seq2[curr_pos:end_pos]
            file.write(f"{name1.ljust(10)} {block1}\n")
            file.write(f"{name2.ljust(10)} {block2}\n")
            file.write("\n")  # Add a blank line between blocks
            # Update current position
            curr_pos = end_pos
    return

def write_fastas_to_directory(output_df, output_directory = ""):
    '''Repeatedly calls the write_fasta function to a directory through a DataFrame.'''
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory) # makes directory if it doesn't exist

    for _, row in output_df.iterrows():
        name, sequence = row["Entry Name"], row["Sequence"]
        description = f" | {row["Number of Repeats"]} Repeats Found | Residues {row["Start"]} to {row["End"]}"
        write_fasta(name, sequence, row["Repeat Locations"], description, output_name = f"{output_directory}{name}_output.fasta")
    return

def write_alns_to_directory(amino_df, embed_df, output_directory = ""):
    '''Repeatedly calls the write_aln function to a directory, using two DataFrames whose items are aligned based on matching names.'''
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory) # makes directory if it doesn't exist
    amino_df = amino_df.rename(columns = {"Sequence":"Amino Sequence"})
    embed_df = embed_df.rename(columns = {"Sequence":"Embed Sequence"})
    combined_df = amino_df.merge(embed_df, on = "Entry Name", how = "outer") # renames columns and combines based on Entry Name

    for _, row in combined_df.iterrows():
        name, amino_sequence, embed_sequence, break_indexes = row["Entry Name"], row["Amino Sequence"], row["Embed Sequence"], row["Repeat Locations"]
        write_aln(f"{name}_AMINO", f"{name}_EMBED", amino_sequence, embed_sequence, break_indexes, output_name = f"{output_directory}{name}_output.aln")
    return

def write_tsv(output_df, output_name = "output.tsv"):
    '''Writes a single TSV files from a DataFrame'''
    output_df.to_csv(output_name, sep = "\t", index = False)
    return
