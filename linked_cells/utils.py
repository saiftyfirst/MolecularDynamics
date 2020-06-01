import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product


def lj_potential(distance, c1=1e-15, c2=1e-5):
    return   (c1 / distance**12) - (c2 / distance**6)

def get_successor_neighbor_delta_coordinate(a=1):
    """Returns neighbor_delta_coordinate
    
    Parameters
    ---------
    a: int
        Variable linked-cell parameter
    """
        
    # determine all possible neighbor candidates
    candidates = np.array(list(product(np.arange(1, a + 1), np.arange(1, a + 1))))
    
    # determine which candidates are actually in the radius of the center cell
    cond = [np.sqrt((elem[0] - 1) ** 2 + (elem[1] - 1) ** 2) < a for elem in candidates]

    # append cells that match condition to neighbor_delta_coordinate
    neighbor_delta_coordinate = list(candidates[cond])
    
    return neighbor_delta_coordinate

def plot_all_cells(ax, list_cells, edgecolor='r',domain=1):
    for c in list_cells:
        c.plot_cell(ax, edgecolor=edgecolor)
    ax.tick_params(axis='both',labelsize=0, length = 0)
    plt.xlim(left=0, right=domain)
    plt.ylim(bottom=0, top=domain)
    ax.set_aspect('equal', adjustable='box')
    
def get_mean_relative_error(direct_potential, linked_cell_potential):
    return np.mean(np.abs((direct_potential - linked_cell_potential) / direct_potential))
    