"""

**equation.py**

@Author:    Matthew Holland
@Date:      2025-02-18
@Version:   0.0
@Contact:   matthew.holland@my.utsa.edu

    This file contains the equation objects to solve the following partial differential equations:

    - Burgers Equation (w/ & w/out dissipation)
    - Kuramoto-Sivashinsky Equation

Changelog

Version     Date            Author              Notes

0.0         2025-02-14      Matthew Holland     Initial version of the file, imported objects from
                                                    ME 5653 repository, available at: https://github.com/mhland10/me5653_CFD_repo

"""

#==================================================================================================
#
#   Importing Required Modules
#
#==================================================================================================

import numpy as np
from distributedObjects import numericalGradient
import scipy.sparse as spsr

#==================================================================================================
#
#   PDE Equations
#   
#==================================================================================================

class eqn_problem(object):
    """
        This object is the object that is injected into the integrator as a generalized PDE 
    problem.
    
    """
    def __init__(self, spatial_order, spatialBC_order, stepping="explicit", max_derivative=4 ):
        """
            Initialize the discretized PDE problem.

        Args:
            spatial_order (int, optional):  The theoretical order that the spatial gradient will be
                                                calculated by, i.e. the number of points in the 
                                                stencil. Defaults to 2.

            spatialBC_order (int, optional):    The theoretical order that the spatial gradient
                                                    will be calculated by at the boundary 
                                                    conditions. Defaults to None, which sets the 
                                                    value to "spatial_order".

            stepping (str, optional):   The stepping method that will be used to solve the PDE.
                                            Defaults to "explicit", or can be implicit. Not case
                                            sensitive.
        """

        # Check stencil
        if spatial_order < max_derivative:
            raise Warning("Spatial order is less than the maximum derivative order. This may cause issues.")

        # Set spatial order
        self.spatial_order = spatial_order
        self.spatialBC_order = spatialBC_order

        # Set the spatial gradient object
        self.spatialGradient_dx = numericalGradient(1, (self.spatial_order//2, self.spatial_order//2))
        self.spatialGradient_dx2 = numericalGradient(2, (self.spatial_order//2, self.spatial_order//2))  
        self.spatialGradient_dx3 = numericalGradient(3, (self.spatial_order//2, self.spatial_order//2))  
        self.spatialGradient_dx4 = numericalGradient(4, (self.spatial_order//2, self.spatial_order//2))  

        # Store stepping method
        self.stepping = stepping.lower()      

    def __call__(cls, x, u, coeffs, BC_x, BC_dx, BC_dx2, BC_dx3, BC_dx4):
        print("Hello there")

class burgers_eqn(eqn_problem):
    """
        This object contains the data and methods to solve the Burgers Equation. Inherits from the
    generalized PDE problem object.
    
    """
    def __init__(self, spatial_order=2, spatialBC_order=None, stepping="explicit", viscid=True):
        """
            Initialize the Burgers Equation problem.
        
        """
        # Set up boundary condition order
        if spatialBC_order is None:
            spatialBC_order = spatial_order

        # Initialize from eqn_problem
        if viscid:
            super().__init__(spatial_order, spatialBC_order, stepping=stepping, max_derivative=2)
        else:
            super().__init__(spatial_order, spatialBC_order, stepping=stepping, max_derivative=1)

    def __call__(cls, x, u, nu, BC_x, BC_dx, BC_dx2, *args):
        """
            Set of Differential equations to solve the Burgers Equation.

            The equation is set up in the following format:

        **Explicit methods**

        <du/dt>=[A]<u>+[B]<f>+[E]<e>

        Where 
        - <du/dt> is the time derivative of the solution, which is the output
        - [A] is the matrix of spatial derivative operators for the function <u>
        - <u> is the solution to the Burgers Equation in vector corresponding to the mesh
        - [B] is the matrix of spatial derivative operators for the function <f>
        - <f> is the flux form of the solution to the Burgers Equation (f=u^2/2) in vector 
                corresponding to the mesh
        - [E] is the matrix of terms to match the boundary conditions, e to the discretized
                equation.
        - <e> is the boundary conditions.

        Args:
            x (np.ndarray):         The spatial grid

            u (np.ndarray):         The function of the Burgers Equation

            BC_x (np.ndarray):      The boundary conditions for the spatial grid

            BC_dx (np.ndarray):     The boundary conditions for the spatial derivative
        
        """

        # Set up A-matrix
        A = cls.spatialGradient_dx2.formMatrix(len(x))

        # Set up B-matrix
        B = cls.spatialGradient_dx.formMatrix(len(x))
        f = u*u/2

        #
        # Set up BCs
        #

        # Set A & B matrix entries to zero as appropriate
        A = A.tocsr()
        B = B.tocsr()
        if BC_x[0] or BC_dx[0]:
            A[0,0]= 0
            B[0,0]= 0
        if BC_x[1] or BC_dx[1]:
            A[-1,-1]= 0 
            B[-1,-1]= 0

        # Set up E-matrix and e-vector as appropriate
        E = spsr.csr_matrix((len(x), len(x)))
        e = np.zeros(len(x))
        if BC_x[0]:
            E[0,0] = 1
            e[0] = BC_x[0]
        if BC_dx[0]:
            E[0,0] = 1
            E[0,1] = -1
            e[0] = BC_dx[0]
        if BC_x[1]:
            E[-1,-1] = 1      
            e[-1] = BC_x[1]
        if BC_dx[1]:
            E[-1,-1] = 1
            E[-1,-2] = -1
            e[-1] = BC_dx[1]

        # Sum to time derivative
        du_dt = A.dot(u) + B.dot(f) + E.dot(e)




