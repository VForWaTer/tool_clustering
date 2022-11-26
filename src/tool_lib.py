from typing import Union
from sklearn import cluster
import pandas as pd
import numpy as np

def init_cluster(method: str, n_clusters: int = None, random_state: int = 42,  **args):
    if method.lower() == 'kmeans':
        return cluster.KMeans(n_clusters=n_clusters, random_state=random_state, **args)
    elif method.lower() == 'mean_shift':
        return cluster.MeanShift(**args)
    elif method.lower() == 'affinity_propagation':
        return cluster.AffinityPropagation(random_state=random_state, **args)
    elif method.lower() == 'agglomerative_single':
        return cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='single', **args)
    elif method.lower() == 'agglomerative_ward':
        return cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='single', **args)
    elif method.lower() == 'agglomerative_complete':
        return cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='single', **args)
    elif method.lower() == 'agglomerative_average':
        return cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='single', **args)
    raise ValueError(f"The method '{method}' is not supported")


def parse_data(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    # use only the data array
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # handle dimensionality
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # flatten dimensionality until it's 2 dimensional
    did_flatten = False
    while data.ndim > 2:
        did_flatten = True
        data = data.reshape(-1, data.shape[-1])
    
    if did_flatten:
        print("You can only pass a matrix of shape (n_samples, n_features). Your data was higher-dimensional. The data was flattened, which is likely not what you are after.")
    
    return data


def get_results(cl, data: np.ndarray, center_by = 'mean'):
    # get the labels, these are always present
    labels = cl.labels_

    # check if cluster centers are present
    if hasattr(cl, 'cluster_centers_'):
        centers = cl.cluster_centers_
    else:
        # create a DataFrame and group
        df = pd.DataFrame(data).groupby(labels)
        if center_by == 'mean':
            centers = df.mean().values
        elif center_by == 'median':
            centers = df.median().values
        else:
            f = getattr(df, center_by)
            centers = f().values
    
    # return labels and centers
    return labels, centers
