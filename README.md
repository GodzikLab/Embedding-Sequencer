# PLM Embedding-Sequencer
 
## Overview

This script uses the clustering of  protein language model embeddings to generate alternative sequences that better reflect the protein's biological features. These sequences can be used to rapidly identify structural repeats and other protein domains.

We are currently using the **ESM-2 35M model.**

## Usage

The script takes 3 inputs and has 2 optional flags to be used like so:
```
sequencer.py pool.hdf input.fasta output.fasta
```
```
sequencer.py -s -p pool.hdf input.fasta output.fasta
```

Included Components:
* requirements.txt - Used for setting up the virtual environment.

* HDF file(s) - Used to designate the <ins>clustering pool</ins> in use.

* sequencer.py script - Self-explanatory.

## Key Components / Explanation

### HDF Files & Clustering Pool

In order to cluster the embeddings into letters, the method uses a <ins>clustering pool</ins> of preselected proteins with locations of interest. The embeddings of the pool are clustered into letters, optimized to identify a repeating pattern at each of the locations of interest.

For example, the hTLR pool is optimized to find leucine-rich repeats. It has 7 letters/clusters and uses the 'FEDED' pattern to identify repeats in the alternate sequence. This information is stored as an HDF file to avoid reclustering the pool for each instance of use and to maintain consistency across the letters.

### The Sequence Flag (-s)

As mentioned, the script can find locations of interests in the protein sequence using the pattern defined in the HDF file. However, if one wanted to ignore the find function and only output the sequence. You can use the '-s' flag in the command.

### The Print Flag (-p)

The print flag can be used for debugging or tracking the script's behavior as it runs. It is generally unnecessary for regular use.

## Dependencies

- **PyTorch** - For ESM-2 embedding generation  
- **Biopython** - For manipulating FASTA files  
- **h5py** - For manipulating HDF5 files  
- **FAISS** - For efficient similarity search
- **sci-kit learn** - For preprocessing of embeddings