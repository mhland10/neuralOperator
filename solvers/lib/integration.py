"""
**integration.py**

@Author:    Matthew Holland
@Contact:   matthew.holland@my.utsa.edu
@Date:      2025/02/18
@Version:   0.0

"""

#==================================================================================================
#
#   Importing Required Modules   
#
#==================================================================================================

import numpy as np

#==================================================================================================
#
#   Integration Objects
#
#==================================================================================================

class explicitEuler:
    """
        This object contains the data and methods to perform an explicit Euler integration scheme. 
    May also be called Lax method.

    """
    def __init__(self):
        print("Initializing explicit Euler integrator...")

    def __call__(cls, eqn, x, u_0, dt, C, nu=0.0, mu=0.0, gamma=0.0, BC_x=None, BC_dx=None, BC_dx2=None, BC_dx3=None, BC_dx4=None):