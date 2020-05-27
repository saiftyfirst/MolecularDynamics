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
    
    neighbor_delta_coordinate = []
    ############# Task 1.1 begins ##################

    # Cells are quadratic and their side-length is by definition
    # The cutoff radius is not necessary since it cancels out
    # cell_side_length = 

    # We have to cyclicly considere more and more layers around the center cell
    for x_offset_index in range(1, a+1):
        for y_offset_index in range(1, a+1):
            # We have to reduce the index by one since if a cell would partially
            # lie in the cut-off radius then its point closest to the top right
            # corner of the centered cell must be above its bottom left corner
            if np.sqrt((x_offset_index - 1)**2 + (y_offset_index - 1)**2) < a:
                neighbor_delta_coordinate.append(
                    np.array([x_offset_index, y_offset_index])
                )
    ############ Task 1.1 ends #####################
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
