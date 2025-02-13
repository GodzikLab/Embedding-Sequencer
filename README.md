# PLM Embedding-Sequencer
 
## Overview

This script uses the clustering of  protein language model embeddings to generate sequences that use a cluster-based alphabet to better reflect the protein's biological features. These sequences can be used to rapidly identify structural repeats and other protein domains by identifying patterns present in protein language models. 

We are currently using the **ESM-2 35M model.**

## Usage

### Recommended Initial Usage:

```
python main.py file_input/Q6PEZ8.fasta -f -p
```
OR
```
python main.py file_input/example.tsv -t -p
```

The script has 2 mandatory parameters: the input and the output type (selected by a flag)
```
sequencer.py input -output_flag
```
```
main.py [-h] input_path (-f | -a | -t) [--hdf HDF] [--output OUTPUT] [-p]
```
* Enter an input as a FASTA, directory of FASTAs, or CSV/TSV file. 
* Choose output type (f - FASTA, a - ALN, t - TSV) and whether or not to show print statements (-p)

## Key Components / Explanation

### HDF Files

In order to cluster the embeddings into letters, the method uses a <ins>clustering pool</ins> of preselected proteins with locations of interest. The embeddings of the pool are clustered into letters, optimized to identify a repeating pattern at each of the locations of interest.

For example, the default hTLR pool is optimized to find leucine-rich repeats. It has 7 letters/clusters and uses the 'FEDED' pattern to identify repeats in the cluster-label sequence. The embeddings, clustering information, and pattern is stored in the HDF file to avoid reclustering the pool for each instance of use and to maintain consistency across the letters.

### False Positive Avoidance

The current iteration of the project uses only identifies Leucine-Rich Repeats if there are more than 2 instances of the repeating pattern. If there are only 2 instances, none of the repeats will be highlighted and the protein will be determined as a "non-repeat."

### The Print Flag (-p)

The print flag can be used for debugging or tracking the script's behavior as it runs. It is generally unnecessary for regular use.

## Dependencies

- **PyTorch** - For ESM-2 embedding generation  
- **Biopython** - For manipulating FASTA files  
- **h5py** - For manipulating HDF5 files  
- **FAISS** - For efficient similarity search
- **sci-kit learn** - For preprocessing of embeddings