#!/bin/bash

# directories
INPUT_DIR="fasta_input"
OUTPUT_DIR="fasta_output"

# hdf file
HDF_FILE="20241205_hTLR_pool.hdf"

# flags
FLAG1="-f"

# make output directory if missing
mkdir -p "$OUTPUT_DIR"

# apply python script to each file in the input folder
for INPUT_FILE in "$INPUT_DIR"/*; do 
    # extract base name of input file
    BASENAME=$(basename "$INPUT_FILE")
    OUTPUT_BASENAME="${BASENAME%.*}_sequenced.${BASENAME##*.}"

    # define the output file name in the output folder
    OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_BASENAME"

    # run the python script with flags and i/o file names
    python sequencer.py $FLAG1 "$HDF_FILE" "$INPUT_FILE" "$OUTPUT_FILE"

    # print progress
    echo "Processed: $INPUT_FILE -> $OUTPUT_FILE"
done