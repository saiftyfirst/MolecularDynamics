import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches


#----- class Point definition-----#
class Point():
    
    """The class for a point.
    
    Arguments:
        coords: a two-element list, containing the 2d coordinates of the point.
        domain: the domain of random generated coordinates x,y default=1.0.
    
    Attributes:
        x, y: coordinates of the point.
    """
    
    def __init__(self, coords=[], domain=1.0):
        if coords:
            assert len(coords) == 2, "the size of coords should be 3."
            self.x = coords[0]
            self.y = coords[1]
        else:
            self.x = domain * np.random.random()
            self.y = domain * np.random.random()
            
    def distance(self, other):
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)


class Particle(Point):
    
    """The derived class for a particle, inheriting the base class "Point".
    
    Attributes:
        m: mass of the particle.
        phi: the potential of the particle.
    """
    
    def __init__(self, coords=[], domain=1.0, m=1.0):
        Point.__init__(self, coords, domain)
        self.m = m
        self.phi = 0.
        
    def plot(self, marker='o', color='r', s=2):
        plt.scatter(self.x, self.y, marker=marker, color=color, s=s)