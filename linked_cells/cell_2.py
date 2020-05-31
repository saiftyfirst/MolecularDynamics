import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product
import particle as pr
import utils
from cell import *


class Cell_2(Cell):
    
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
        Cell.__init__(self, lx, ly, r_c, cell_index, neighbor_delta_coordinate, a=a, domain=domain)
        self.particle_list = []
       
    def p2p_self(self):
        """calculates the potential on all particle inside a cell due to particles in the same cell
        """
        ############## Task 5.1 begins ################

        for particle_id, particle in enumerate(self.particle_list):
            for other_id, other in enumerate(self.particle_list):
                if(particle_id != other_id):
                    distance = particle.distance(other)
                    particle.phi += utils.lj_potential(distance)

        ############## Task 2.1 ends ################
    
    def p2p_neigbor_cells(self, list_cells):
        """calculates the potential on all particle inside a cell due to particles in the neighor cells
        
        Parameters
        ----------
        list_cells: list
            List of all cells
        """
        ############## Task 5.2 begins ################

        for particle in self.particle_list:
            for neighbor in self.neighbor_cell_index:
                for other in list_cells[neighbor].particle_list:
                    distance = particle.distance(other)
                    particle.phi += utils.lj_potential(distance)

        ############## Task 5.2 ends ################
                
    def calculate_potential(self, list_cells):
        """calculates the potential on all particle inside a cell
        
        Parameters
        ----------
        list_cells: list
            List of all cells
        """
        ############## Task 5.3 begins ################

        self.p2p_self()
        self.p2p_neigbor_cells(list_cells)

        ############## Task 5.3 ends ################
            
    def add_particle(self, particle):
        """Append particle index at end of list_particles
        
        Parameters
        ----------
        particle: objectof type Particle
            Particle to be added in the cell
        """
        self.particle_list.append(particle)
        
    def add_neighbor_cell(self, cell_index):
        self.list_interaction_cells.append(cell_index)
        
    def delete_all_particles(self):
        self.particle_list = []
      
    def plot_particles(self, marker='o', color='r', s=2):
        for p in self.particle_list:
            p.plot(marker=marker, color=color, s=s)
            
    def plot_neighbor_cell_particles(self, list_cells, marker='o', color='r', s=2):
        for c_idx in self.neighbor_cell_index:
            list_cells[c_idx].plot_particles(marker=marker, color=color, s=s)        

        
def create_assign_particle_to_cell(N, list_cells, r_c, domain=1, a=1):
    """Creates and assigns particle to cell
        
    Parameters
    ----------
    list_cells: list
        List of all cells
    r_c: float
        Cut-off radius
    domain: float
        Size of domain
    a: int
    """
    side_length = r_c / a
    n = np.int(np.ceil(domain / side_length))
    for idx in range(N):
        particle = pr.Particle(domain=domain)
        pos_x, pos_y =  int(np.floor(particle.x / side_length)), int(np.floor(particle.y / side_length))
        list_cells[pos_y * n + pos_x].add_particle(particle)
    
def extract_linked_cell_potential(N, list_cells):
    potential = np.zeros(N, dtype=np.float)
    idx = 0
    for cell in list_cells:
        for p in cell.particle_list:
            potential[idx] = p.phi
            idx = idx + 1
    return potential

def direct_interaction_one_particle_v2(particle, list_cells):
    potential = 0
    for cell in list_cells:
        for p in cell.particle_list:
            distance = particle.distance(p)
            if distance > 1e-10:
                potential += utils.lj_potential(distance)
    return potential

def direct_interaction_v2(N, list_cells):
    potential = np.zeros(N, dtype=np.float)
    idx = 0
    for cell in list_cells:
        for p in cell.particle_list:
            potential[idx] = direct_interaction_one_particle_v2(p, list_cells)
            idx = idx + 1
    return potential

def get_list_cell(r_c, neighbor_delta_coordinate, domain=1.0, a=1):
    """Return list containing cells in row major format
    
    Parameters:
    ----------
    r_c : float
        Cut-off radius 
    neighbor_delta_coordinate: list
        List of  List of numpy array(of size 2), where each numpy array contains the difference between the 
        2d coordinates of current cell and one of neighbor cells in upper right quadrant
    domain : float (Default value: 1.0)
        Domain size
    a : int (Default value: 1.0)
        Variable linked cell parameter
        
    Return
    ------
    list :  List of cells in row major format
    """
    list_cells = []
    side_length = r_c / a   
    n = np.int(np.ceil(domain / side_length))
    cell_idx = 0
    for y in range(n):
        for x in range(n):
            temp_cell = Cell_2(x * side_length, y * side_length, r_c, cell_idx, neighbor_delta_coordinate, 
                               a=a, domain=domain)
            list_cells.append(temp_cell)
            cell_idx = cell_idx + 1
    return list_cells

def calculate_linked_cell_potential(list_cells):
    for cell in list_cells:
        for p in cell.particle_list:
            p.phi = 0.
        cell.calculate_potential(list_cells)
