import os
from datetime import datetime as dt
import time
import json

from json2args import get_parameter
import numpy as np

from tool_lib import init_cluster, parse_data, get_results

# parse parameters
kwargs = get_parameter()

# check if a toolname was set in env
toolname = os.environ.get('TOOL_RUN', 'cluster').lower()

# switch the tool
if toolname == 'cluster':
    # get the parameter
    try:
        data = parse_data(kwargs['data'])
    except Exception as e:
        print(str(e))
        raise e
        
    # initialize the cluster instance
    try:
        cl = init_cluster(
            method=kwargs['method'],
            n_clusters=kwargs.get('n_clusters'),
            random_state=kwargs.get('random_state', 42),
            **kwargs.get('init_args', {})
        )
    except KeyError as e:
        print("Mandatory data is missing, please check the Tool specification.")
        raise e

    # run the Algorithm
    t1 = time.time()
    cl.fit(data)
    t2 = time.time()
    print(f"Clustering took {t2 - t1} seconds")

    # get results
    labels, centers = get_results(cl, data)

    # save the results
    np.savetxt('/out/labels.dat', labels, fmt="%d")
    np.savetxt('/out/cluster_centers.dat', centers)
    with open('/out/clustering.json', 'w') as f:
        json.dump(dict(labels=labels, centers=centers), indent=4)


# In any other case, it was not clear which tool to run
else:
    raise AttributeError(f"[{dt.now().isocalendar()}] Either no TOOL_RUN environment variable available, or '{toolname}' is not valid.\n")
