import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product
import utils
from abc import ABCMeta, abstractmethod


class Cell(object, metaclass=ABCMeta):
    
    def __init__(self, lx, ly, r_c, cell_index, neighbor_delta_coordinate, a=1, domain=1):
        """Constructor
        lx : float
            Lower x coordinate
        ly : float
            Lower y coordinate
        r_c : float
            Cut-off radius
        cell_index : int
            Index of cell in list_cells
        neighbor_delta_coordinate : list
            List of  List of numpy array(of size 2), where each numpy array contains the difference between the 
            2d coordinates of current cell and one of neighbor cells in upper right quadrant
        a : int (default value 1)
            Variable lined-cell parameter
        domain : float (default value 1.0)
            Size of domain
        """
        self.side_length = r_c / a
        self.a = a
        self.cell_center = np.array([lx + 0.5 * self.side_length, ly + 0.5 * self.side_length])
        self.cell_index = cell_index
        self.neighbor_cell_index = []
        self.create_neighbor_cell_index(neighbor_delta_coordinate, domain=domain)
                
    def create_neighbor_cell_index(self, neighbor_delta_coordinate, domain=1):
        """Creates neighbor cell index for the current cell
        Parameters
        ----------
        neighbor_delta_coordinate: list
            Relative 2d index of all neighbor interaction cells in first quadrant
        domain: float (Optional value 1.0)
            Size of domain
        """
        self.neighbor_cell_index = []
        ############## Task 1.2 begins ##################

        side_cells = int(np.ceil(domain / self.side_length))
        cell_i = self.cell_index % side_cells
        cell_j = int((self.cell_index - cell_i) / side_cells)

        for offset in neighbor_delta_coordinate:
            # Quadrant 1
            neighbor_i = cell_i + offset[0]
            neighbor_j = cell_j + offset[1]
            if (neighbor_i < side_cells) and (neighbor_j < side_cells):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)
            # Quadrant 2
            neighbor_i = cell_i - offset[0]
            neighbor_j = cell_j + offset[1]
            if (neighbor_i > -1) and (neighbor_j < side_cells):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)
            # Quadrant 3
            neighbor_i = cell_i - offset[0]
            neighbor_j = cell_j - offset[1]
            if (neighbor_i > -1) and (neighbor_j > -1):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)
            # Quadrant 4
            neighbor_i = cell_i + offset[0]
            neighbor_j = cell_j - offset[1]
            if (neighbor_i < side_cells) and (neighbor_j > -1):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)

        for i in range(1, self.a+1):
            # Quadrant 1
            neighbor_i = cell_i + i
            neighbor_j = cell_j
            if (neighbor_i < side_cells):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)
            # Quadrant 2
            neighbor_i = cell_i - i
            neighbor_j = cell_j
            if (neighbor_i > -1) and (neighbor_j < side_cells):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)
            # Quadrant 3
            neighbor_i = cell_i
            neighbor_j = cell_j - i
            if (neighbor_i > -1) and (neighbor_j > -1):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)
            # Quadrant 4
            neighbor_i = cell_i
            neighbor_j = cell_j - i
            if (neighbor_i < side_cells) and (neighbor_j > -1):
                self.neighbor_cell_index.append(neighbor_i + side_cells*neighbor_j)


        ############## Task 1.2 ends ##################
        
    def __str__(self):
        return 'Object of type cell with center {}'.format(self.cell_center)
    
    @abstractmethod
    def add_particle(self, particle_index):
        return
        
    def add_neighbor_cell(self, cell_index):
        self.neighbor_cell_index.append(cell_index)
    
    @abstractmethod
    def delete_all_particles(self):
        return
          
    def plot_cell(self, ax, linewidth=1, edgecolor='r', facecolor='none'):
        lx = self.cell_center[0] - self.side_length/2
        ly = self.cell_center[1] - self.side_length/2
        rect = patches.Rectangle((lx, ly), self.side_length, self.side_length, linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
   
    @abstractmethod
    def plot_particles(self, list_particles, marker='o', color='r', s=2):
        return
            
    def plot_neighbor_cells(self, ax, list_cells, linewidth=1, edgecolor='r', facecolor='none'):
        for idx in self.neighbor_cell_index:
            list_cells[idx].plot_cell(ax, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
            
    @abstractmethod
    def plot_neighbor_cell_particles(self, list_cells, list_particles, marker='o', color='r', s=2):
        return
    
    def distance(self, other):
        return np.linalg.norm(self.cell_center - other.cell_center, 2)
    
    def plot_rc(self, ax, rc):
        circle = patches.Circle((self.cell_center[0], self.cell_center[1]), rc)
