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

            max_derivative (int, optional): The maximum derivative that will be calculated in space
                                                for the equation. Will come from the equation 
                                                object. Default value is 4.

        Attributes:
            spatial_order <= spatial_order

            spatialBC_order <= spatialBC_order

            spatialGradients (numericalGradients object):   The list of numericalGradients objects
                                                                that correspond to the i+1 spatial
                                                                derivative.

            stepping <= stepping
            
        """

        # Check stencil
        if spatial_order < max_derivative:
            raise Warning("Spatial order is less than the maximum derivative order. This may cause issues.")

        # Set spatial order
        self.spatial_order = spatial_order
        self.spatialBC_order = spatialBC_order

        # Set the spatial gradient object list
        self.spatialGradients = []
        for i in range( max_derivative ):
            self.spatialGradients += [numericalGradient(i+1, (self.spatial_order//2, self.spatial_order//2))]

        # Store stepping method
        self.stepping = stepping.lower()      

    def __call__(cls, x, u, coeffs, BC_x=( None, None ), BC_dx=( None, None ), BC_dx2=( None, None ), BC_dx3=( None, None ), BC_dx4=( None, None ) ):
        """
            This method is what happens when the eqn_problem object is called by some function or 
        such. It is set up to accept a 1D space, parameter, coefficients to the equations, and 
        boundary conditions.

            Note that the boundary conditions must correspond to the specific equation that is 
        being used.

            This method is what sets up the boundary conditions in the terms for the spatial 
        gradients and the E<e> term which contains the 

        Args:
            x (numpy 1darray - float):  The 1D spatial domain. 

            u (numpy 1darray - float):  The function as it correlates to "x".

            coeffs (numpy 1darray - float): The coefficients that pertain to the spatial gradients
                                                in the equation that is being solved. Refer to the 
                                                specific equation. Can also be a list or tuple.

            BC_x (tuple - float):   The boundary conditions for "u".

            BC_dx (tuple - float):  The boundary conditions for d/dx("u").

            BC_dx2 (tuple - float): The boundary conditions for d^2/dx^2("u").

            BC_dx3 (tuple - float): The boundary conditions for d^3/dx^3("u").

            BC_dx4 (tuple - float): The boundary conditions for d^4/dx^4("u").

        """
        
        # Set up domain
        cls.x = x
        cls.u = u
        cls.dx = np.gradient( cls.x )

        # Apply mesh spacing to spatial gradients
        cls.gradient_matrices = []
        for i in range( len( cls.spatialGradients ) ):
            cls.spatialGradients[i].formMatrix( len( cls.x ) )
            # This multiplies the gradient matrix
            cls.gradient_matrices += [ cls.spatialGradients[i].gradientMatrix.multiply( cls.dx**-(i+1) ) ]

        #
        # Set up boundary conditions
        #
        bc_count = 0
        cls.E = spsr.csr_matrix( ([], ([], [])), shape= (len(x), len(x)) )
        cls.e = np.zeros( len(x) )
        for i in range( len( cls.gradient_matrices ) ):
            cls.gradient_matrices[i] = cls.gradient_matrices[i].tocsr()
        
        # BC LHS - x
        if not BC_x[0]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_x[0]=="same":
                cls.E[0,0]=1
                cls.e[0]=BC_x[0]

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][0,:] = np.zeros_like( cls.gradient_matrices[i][0,:] )

        # BC RHS - x
        if not BC_x[-1]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_x[-1]=="same":
                    cls.E[-1,-1]=1
                    cls.e[-1]=BC_x[-1]

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][-1,:] = np.zeros_like( cls.gradient_matrices[i][-1,:] )
        
        # BC LHS - dx
        if not BC_dx[0]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_dx[0]=="same":
                cls.E[0,:]=cls.gradient_matrices[0][0,:]
                cls.e[0]=BC_dx[0]

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][0,:] = np.zeros_like( cls.gradient_matrices[i][0,:] )
            
        # BC RHS - dx
        if not BC_dx[-1]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_dx[-1]=="same":
                cls.E[-1,:]=cls.gradient_matrices[0][-1,:]
                cls.e[-1]=BC_dx[-1]

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][-1,:] = np.zeros_like( cls.gradient_matrices[i][-1,:] )
            
        # BC LHS - dx2
        if not BC_dx2[0]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_dx2[0]=="same":
                cls.E[0,:]=cls.gradient_matrices[0][0,:]
                cls.e[0]=BC_dx2[0]

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][0,:] = np.zeros_like( cls.gradient_matrices[i][0,:] )
            
        # BC RHS - dx2
        if not BC_dx2[-1]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_dx2[-1]=="same":
                cls.E[-1,:]=cls.gradient_matrices[0][-1,:]
                cls.e[-1]=BC_dx2[-1]

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][-1,:] = np.zeros_like( cls.gradient_matrices[i][-1,:] )

        # TODO: Add in higher order derivatives as needed   

        cls.bc_count = bc_count 
        cls.E = cls.E.todia() 

        

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
        print(f"Spatial order is {spatial_order}")

        # Initialize from eqn_problem
        if viscid:
            super().__init__(spatial_order, spatialBC_order, stepping=stepping, max_derivative=2)
        else:
            super().__init__(spatial_order, spatialBC_order, stepping=stepping, max_derivative=1)
        self.viscid=viscid

    def __call__(cls, x, u, coeffs, BCs, *args):
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

        nu=coeffs[0]

        # Pull the boundary conditions
        BC_x = BCs[0]
        BC_dx = BCs[1]

        # Call the parent class call method
        if cls.viscid:
            super().__call__(x, u, nu, BC_x=BC_x, BC_dx=BC_dx, BC_dx2=BC_dx2 )
        else:
            super().__call__(x, u, nu, BC_x=BC_x, BC_dx=BC_dx )

        # Set up A-matrix - ie: 2nd derivative
        if cls.viscid and not nu[0]==0:
            cls.A = nu[0] * cls.gradient_matrices

        # Set up B-matrix
        cls.B = cls.gradient_matrices[0].todia()
        cls.f = u*u/2

        # Sum to time derivative
        du_dt = -cls.B.dot(cls.f) + cls.E.dot(cls.e)
        if cls.viscid and not nu[0]==0:
            du_dt += -cls.A.dot(u)

        return du_dt




