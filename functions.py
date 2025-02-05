import numpy as np

import h5py # working with HDF files

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






