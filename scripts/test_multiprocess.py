# %%
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
import igraph as ig

g = ig.Graph(directed=True)
origin_dest_dict = {"A": "D", "B": "E", "C": "F"}


# Function to find the shortest path
def get_shortest_path(params):
    origin, destination = params
    path = g.get_shortest_paths(origin, to=destination, output="vpath")[0]
    return (origin, path)


if __name__ == "__main__":
    origins_destinations = [
        (origin, destination) for origin, destination in origin_dest_dict.items()
    ]

    # Using Pool for multiprocessing
    with Pool(processes=4) as pool:
        results = pool.map(get_shortest_path, origins_destinations)

    # Process results
    shortest_paths = {origin: path for origin, path in results}
    print(shortest_paths)


# %%
args = [1, 2, 3, 4, 5]

"""multiprocesses:
https://medium.com/@mehta.kavisha/different-methods-of-multiprocessing-in-python-70eb4009a990
# Pool function requires the __main__ module to be importable by the children.
This means multiprocessing.Pool will not work in the interactive interpreter.

"""

if __name__ == "__main__":
    with Pool(processes=2) as pool:
        results = pool.starmap(function, args)
    print(results)
