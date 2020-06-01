import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product
import utils
from cell import *


class Cell_1(Cell):
    
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
        self.particle_index = []
        
    def p2p_self(self, list_particles):
        """calculates the potential on all particle inside a cell due to particles in the same cell
        
        Parameters
        ----------
        list_particles: list
            List of all "Particle" objects
        """
        # filter out particles in current cell
        cell_particles = np.array(list_particles)[self.particle_index]
        
        # iterate over all particles and calulcate the aggregate potential
        for index, particle in enumerate(cell_particles):
            # slice list to avoid double calculation
            for neighbor in cell_particles[(index + 1):]:
                distance = particle.distance(neighbor)
                phi = utils.lj_potential(distance)
                particle.phi += phi
                neighbor.phi += phi
            
    def p2p_neigbor_cells(self, list_particles, list_cells):
        """calculates the potential on all particle inside a cell due to particles in the neighor cells
        
        Parameters
        ----------
        list_particles: list
            List of all "Particle" objects
        list_cells: list
            List of all cells
        """
        # convenicence with np
        all_particles = np.array(list_particles)
        current_cell_particles = all_particles[self.particle_index]
        
        # filter out cells that are neighbors to current cell
        neighbor_cells = np.array(list_cells)[self.neighbor_cell_index]
        
        # loop through each neighboring cell
        for neighbor_cell in neighbor_cells:
            
            # list of all neighbor particles
            neighbor_particles = all_particles[neighbor_cell.particle_index]
            
            # for each paricle in neighbor cell
            for neighbor_particle in neighbor_particles:
                
                # calculate potenital on each particle of current cell
                for current_cell_particle in current_cell_particles:
                    distance = current_cell_particle.distance(neighbor_particle)
                    current_cell_particle.phi += utils.lj_potential(distance)
                
    def calculate_potential(self, list_particles, list_cells):
        """calculates the potential on all particle inside a cell
        
        Parameters
        ----------
        list_particles: list
            List of all "Particle" objects
        list_cells: list
            List of all cells
        """
        self.p2p_self(list_particles)
        self.p2p_neigbor_cells(list_particles, list_cells)
            
    def add_particle(self, particle_index):
        """Append particle index at end of list_particles
        
        Parameters
        ----------
        particle_int: int
            Index of particle to be added in the cell
        """
        self.particle_index.append(particle_index)
        
    def delete_all_particles(self):
        self.particle_index = []
        
    def plot_particles(self, list_particles, marker='o', color='r', s=2):
        for idx in self.particle_index:
            list_particles[idx].plot(marker=marker, color=color, s=s)
            
    def plot_neighbor_cell_particles(self, list_cells, list_particles, marker='o', color='r', s=2):
        for c_idx in self.neighbor_cell_index:
            list_cells[c_idx].plot_particles(list_particles, marker=marker, color=color, s=s)

        
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
    side_length = r_c / a
    cells_per_axis = int(np.ceil(domain / side_length))

    
    # determine lower left coordinates of all cells
    axis_points = np.arange(0, domain, side_length)
    cell_coordinates = np.array(list(product(axis_points, axis_points)))
    
    # use coordinates list to create list of cells
    list_cells = [Cell_1(
        lx = point[1],
        ly = point[0],
        r_c = r_c,
        cell_index = int(point[1] * cells_per_axis + point[0] * cells_per_axis ** 2),
        neighbor_delta_coordinate = neighbor_delta_coordinate,
        a = a,
        domain = domain
    ) for point in cell_coordinates]
    
    return list_cells

def assign_particle_to_cell(list_particles, list_cells, r_c, domain=1, a=1):
    """Assigns particle to cell
        
    Parameters
    ----------
    list_particles: list
        List of all "Particle" objects
    list_cells: list
        List of all cells
    r_c: float
        Cut-off radius
    domain: float
        Size of domain
    a: int
        Variable linked cell parameter
    """
    side_length = r_c / a
    cells_per_axis = int(np.ceil(domain / side_length))
    
    for index, particle in enumerate(list_particles):
        
        # calculate cell index
        x_offset = int(np.floor(particle.x / side_length))
        y_offset = int(np.floor(particle.y / side_length))
        cell_index = x_offset + y_offset * cells_per_axis
        
        # assign particle to cell
        list_cells[cell_index].add_particle(index)
        
def check_particle_assignment(N, list_cells):
    for cell in list_cells:
        assert (np.array(cell.particle_index, dtype=np.int) < N).all()

def set_potential_zero(list_particles):
    for particle in list_particles:
        particle.phi = 0.
        
def calculate_potential_linked_cell(list_cells, list_particles):
    for cell in list_cells:
        cell.calculate_potential(list_particles, list_cells)

def direct_potential_one_particle(idx, list_particles):
    p1 = list_particles[idx]
    potential = 0.
    for p2 in list_particles:
        distance = p1.distance(p2)
        if distance > 1e-10:
            potential += utils.lj_potential(distance)
    return potential

def direct_potential_all_particles(list_particles):
    N = len(list_particles)
    potential = np.zeros(N, dtype=np.float)
    for n in range(N):
        potential[n] = direct_potential_one_particle(n, list_particles)
    return potential

def extract_linked_cell_potential(list_particles):
    N = len(list_particles)
    potential = np.zeros(N, dtype=np.float)
    for n in range(N):
        potential[n] = list_particles[n].phi
    return potential
