"""

**solvers.py**

@Author:    Matthew Holland
@Date:      2025-02-14
@Version:   0.0
@Contact:   matthew.holland@my.utsa.edu

    This file contains the solvers to create data from for the cases pertaining to the
following partial differential equations:

    - Burgers Equation (w/ & w/out dissipation)
    - Kuramoto-Sivashinsky Equation

Changelog

Version     Date            Author              Notes

0.0         2025-02-14      Matthew Holland     Initial version of the file, imported objects from
                                                    ME 5653 repository, available at: https://github.com/mhland10/me5653_CFD_repo
                                                    
"""

#==================================================================================================
#
# Importing Required Modules
#
#==================================================================================================

#
# Import Solver Libararies
#
import os, sys
print(f"solvers file:\t{__file__}")
script_dir = os.path.dirname(os.path.realpath(__file__))
lib_dir = os.path.join(script_dir, 'lib')
print(f"Library directory:\t{lib_dir}")
sys.path.append(lib_dir)
from distributedObjects import *
from distributedFunctions import *
from equation import *
from integration import *


import numpy as np
import scipy.special as spsp
import scipy.sparse as spsr
from numba import njit, prange, jit

#==================================================================================================
#
#   New Improved 1D Solver Objects
#
#==================================================================================================

class problem1D:
    """
        This object contains the necessary data and functions to solve a 1D problem. Generalized so 
    that specific problems can inherit the attributes and methods to solve specific PDE's.

    """
    def __init__(self, x, u_0, t_ends, coeffs=[0], dt=None, C=None, time_integrator="lax", spatial_order=2, spatialBC_order=None, BC_x=None, BC_dx=[0,None] ):
        """
            Initialize the 1D problem. For all objects, attributes, and methods that follow, the 
        unit system must be SI or equivalent.

            A Note for the boundary conditions, the boundary conditions will need to be checked by 
        the individual solver that is using this object and attributes.

        TODO: Add BCs for higher order derivatives

        Args:
            x (float):  The array of the spatial domain.

            u_0 (float):    The initial condition of the problem.

            t_ends (float): The end points of the time domain for the problem.

            coeffs (float, optional):   The coefficients preceding the spatial gradients of the 
                                            problem.

            dt (float, optional):   The time step of the problem. Defaults to None.

            C (float, optional):    The Courant number to allow the time step to become. If 
                                        numeric, this overrides the time step if the Courant number
                                        of the time step is too high. Defaults to None.

            time_integrator (str, optional):    The time integration scheme that will be used. 
                                                    Defaults to "lax".

            spatial_order (int, optional):  The theoretical order that the spatial gradient will be
                                                calculated by, i.e. the number of points in the 
                                                stencil. Defaults to 2.

            spatialBC_order (int, optional):    The theoretical order that the spatial gradient
                                                    will be calculated by at the boundary 
                                                    conditions. Defaults to None, which sets the 
                                                    value to "spatial_order".

            BC_x (_type_, optional):    The boundary conditions as a function of x. Defaults to None.

            BC_dx (list, optional):     The boundary condition as a gradient of x. Defaults to 
                                            [0,None].

        Raises:
            
        """

        # Set up the spatial domain
        self.x = x
        self.Nx = len( x )
        self.u_0 = u_0

        # Set up the time domain
        self.t_start = t_ends[0]
        self.t_end = t_ends[-1]
        self.dt = dt

        # Set up the coefficients
        self.coeffs = coeffs
        self.C = C

        # Set up the orders
        self.spatial_order = spatial_order
        if spatial_order:
            self.spatialBC_order = spatialBC_order
        else:
            self.spatialBC_order = spatial_order
        
        # Set up the boundary conditions
        self.BC_x = BC_x
        self.BC_dx = BC_dx

        # Set up the time integrator
        self.time_integrator = time_integrator

    def solve( cls ):
        """
            Solve the 1D problem.

        Args:
            None
        
        """
        print("Solving the 1D problem.")

        # Set up the integrator
        if cls.time_integrator.lower() in ["lax", "euler"]:
            cls.integrator = explicitEuler( cls.eqn, cls.x, cls.u_0, (cls.t_start, cls.t_end), cls.dt, cls.coeffs, cls.BCs, C=cls.C, spatial_order=cls.spatial_order, spatialBC_order=cls.spatialBC_order )

        # Solve the integrator
        cls.integrator.solve()

class burgers1D(problem1D):
    """
        Solve the Burger's equation via an improved 1D solver that allows more flexible methods
    
    """
    def __init__(self, x, u_0, t_ends, nu=0.0, dt=None, C=None, time_integrator="lax", spatial_order=2, spatialBC_order=None, BC_x=None, BC_dx=[0,None] ):
        """
            Initialize the Burger's equation.

            TODO: Are there non-explicit stepping methods..?

        Args:
            x (float):  The array of the spatial domain.

            u_0 (float):    The initial condition of the problem.

            t_ends (float): The end points of the time domain for the problem.

            nu (float, optional):   The diffusivity coefficient or viscosity for the diffusivity 
                                        terms of the problem. Defaults to 0.0.

            dt (float, optional):   The time step of the problem. Defaults to None.

            C (float, optional):    The Courant number to allow the time step to become. If 
                                        numeric, this overrides the time step if the Courant number
                                        of the time step is too high. Defaults to None.

            time_integrator (str, optional):    The time integration scheme that will be used. 
                                                    Defaults to "lax".

            spatial_order (int, optional):  The theoretical order that the spatial gradient will be
                                                calculated by, i.e. the number of points in the 
                                                stencil. Defaults to 2.

            spatialBC_order (int, optional):    The theoretical order that the spatial gradient
                                                    will be calculated by at the boundary 
                                                    conditions. Defaults to None, which sets the 
                                                    value to "spatial_order".

            BC_x (_type_, optional):    The boundary conditions as a function of x. Defaults to None.

            BC_dx (list, optional):     The boundary condition as a gradient of x. Defaults to 
                                            [0,None].

        """

        # Initialize the parent object
        super().__init__(x, u_0, t_ends, coeffs=[nu], dt=None, C=None, time_integrator="lax", spatial_order=2, spatialBC_order=None, BC_x=None, BC_dx=[0,None] )

        # Set up the Burger's equation
        if nu==0:
            self.eqn = burgers_eqn( spatial_order=self.spatial_order, spatialBC_order=spatialBC_order, stepping="explicit", viscid=False )
        else:
            self.eqn = burgers_eqn( spatial_order=self.spatial_order, spatialBC_order=spatialBC_order, stepping="explicit", viscid=True )


        # Set up the boundary conditions
        self.BCs = [ self.BC_x, self.BC_dx ]

#==================================================================================================
#
# Burgers Equation Objects
#
#==================================================================================================

class burgersEquation_og:
    """
    This object allows a user to solve a Burger's equation. See HW3 for more detail.

    """
    def __init__( self , x , u_0 , t_domain , dt=None , C=None , solver="lax" , nu=0.0 ):
        """
        Initialize the Burger's equation object.

        Args:
            x [float]:  [m] The spatial mesh that will be used for the Burger's equation solve.

                        Note as of 2024/10/31:  Must be uniform mesh.

            u_0 [float]:    [m/s] The function values for the Burger's equation solve. Must
                                correspond to the mesh in "x".

            t_domain (float):   The (2x) entry tuple that describes the time domain that the
                                    solve will be preformed over. The entires must be:

                                ( t_start , t_end )

            dt (float, optional):   [s] The uniform time step. Must be numerical value if "C" is
                                        None. Defaults to None.

            C (float, optional):    [m/s] The C factor of the Burger's equation solve. Must be
                                        numerical value if "dt" is None. Defaults to None.

            solver (str, optional): The solver that will be used to solve the Burger's equation.
                                        The valid options are:

                                    *"LAX": Lax method.
                                        
                                    Defaults to "lax". Not case sensitive.

            nu (float, optional):   [m2/s] The dissipation of the Burger's equation. The default
                                        is 0, which will be an inviscid case.

        """

        if not np.shape( x )==np.shape( u_0 ):
            raise ValueError("x and u_0 must be the same shape.")
        
        #
        # Write domain
        #
        self.x = x
        self.Nx = np.shape( x )[0]

        dx_s = np.gradient( self.x )
        ddx_s = np.gradient( dx_s )
        if ( np.sum( ddx_s ) / np.sum( dx_s ) ) > 1e-3 :
            raise ValueError( "x is not uniform enough." )
        else:
            self.dx = np.mean( dx_s )

        #
        # Sort out time stepping & dissipation
        #
        if C:
            self.C = C
            if dt:
                raise ValueError( "S is present along with dt. Problem is overconstrained. Only one of C and dt must be present." )
            else:
                self.dt = self.C * self.dx
        else:
            self.dt = dt
            self.C = self.dt / self.dx

        #
        # Set up time domain
        #
        self.t = np.arange( t_domain[0] , t_domain[-1] , self.dt )
        self.Nt = len( self.t )

        #
        # Set up the function values
        #
        self.u = np.zeros( ( self.Nt , self.Nx ) )
        self.u[0,...] = u_0

        #
        # Set up solver
        #
        self.solver = solver.lower()
        self.nu = nu
    
    def solve( cls , N_spatialorder=1 , N_timeorder=1 , N_spatialBCorder=None , BC="consistent" ):
        """
        This method solves the Burger's equation for the object according to the inputs 
            to the object and method.

        There are a few things to note with the method. First, the system of equations is
            described as linear equations stored in a diagonal-sparse matrix supplied by SciPy.
            This is done to avoid using extremely large matrices that are stored.

        The system of linear equations can be simply represented as follows:

        [A]<u> = <b> = [C]<v> + [D]<w>

        Here, <v> is the previous time step and <w> is the previous time step squared, in
            accordance tot he flux transfer method.

        This method will march in time

        Args:
            N_spatialorder (int, optional): Spatial order for the solve. Defaults to 1.

            N_timeorder (int, optional):    Time order for the solve. Defaults to 1.

            N_spatialBCorder (int, optional):   The order of the boundary conditions of the 
                                                    spatial gradients. Defaults to None, which 
                                                    makes the boundary conditions gradients the half
                                                    of "N_spatialorder".
        
        """

        #
        # Calculate A matrix
        #
        if cls.solver.lower()=="lax":
            cls.A_matrix = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        #
        # Calculate C matrix
        #
        if cls.solver.lower()=="lax":
            cls.C_matrix = (1/2) * spsr.dia_matrix( ( [ np.ones( cls.Nx ) , np.ones( cls.Nx ) ] , [-1,1] ) , shape = ( cls.Nx , cls.Nx ) )

            if not cls.nu==0:
                cls.visc_grad = numericalGradient( 2 , ( N_spatialorder//2 , N_spatialorder//2 ) )
                cls.visc_grad.formMatrix( cls.Nx )
                cls.C_matrix = cls.C_matrix + cls.nu * cls.visc_grad.gradientMatrix

        #
        # Calculate D matrix
        #
        if cls.solver.lower()=="lax":
            cls.num_grad = numericalGradient( 1 , ( N_spatialorder//2 , N_spatialorder//2 ) )
            cls.num_grad.formMatrix( cls.Nx )
            cls.D_matrix = cls.num_grad.gradientMatrix


            if N_spatialBCorder:
                cls.D_matrix = cls.D_matrix.tolil()

                # The LHS boundary condition
                for i in range( N_spatialorder//2 ):
                    #print("i:{x}".format(x=i))
                    #N_LHS_order = N_spatialorder
                    N_LHS_order = N_spatialBCorder
                    cls.num_grad_LHS = numericalGradient( 1 , ( i , N_LHS_order-i ) )
                    cls.D_matrix[i,:]=0
                    #"""
                    cls.D_matrix[i,i:i+N_LHS_order+1]=cls.num_grad_LHS.coeffs
                    #"""
                    #cls.D_matrix[i,i:(i+N_spatialBCorder+1)]=cls.num_grad_LHS.coeffs

                # The RHS boundary condition 
                for i in range( N_spatialorder//2 ):
                    #N_RHS_order = N_spatialorder
                    N_RHS_order = N_spatialBCorder
                    cls.num_grad_RHS = numericalGradient( 1 , ( N_RHS_order-i , i ) )
                    cls.D_matrix[-i-1,:]=0
                    #"""
                    if i==0:
                        cls.D_matrix[-1,-1-N_RHS_order:]=cls.num_grad_RHS.coeffs
                    else:
                        cls.D_matrix[-i-1,-1-N_RHS_order-i:-i]=cls.num_grad_RHS.coeffs
                    #"""
                    #cls.D_matrix[-i-1,-1-N_RHS_order:]=cls.num_grad_RHS.coeffs
                #"""

                cls.D_matrix = cls.D_matrix.todia()

            cls.D_matrix = -(cls.C) * cls.D_matrix

        #
        # Set up boundary conditions
        #
        if BC.lower()=="consistent":
            cls.C_matrix = cls.C_matrix.tolil()
            cls.D_matrix = cls.D_matrix.tolil()
            cls.C_matrix[0,0]=1
            cls.C_matrix[0,1:]=0
            cls.C_matrix[-1,:]=0
            #cls.C_matrix[-1,-2]=1
            cls.C_matrix[-1,-1]=1
            cls.C_matrix[-1,:] = cls.C_matrix[-1,:].toarray() / np.sum( cls.C_matrix[-1,:].toarray() )
            cls.D_matrix[0,:]=0
            #cls.D_matrix[-1,:]=0
            if not cls.nu==0:
                cls.C_matrix[-1,:]=0
                cls.C_matrix[-1,-1]=1
                cls.D_matrix[-1,:]=0

            cls.C_matrix = cls.C_matrix.todia()
            cls.D_matrix = cls.D_matrix.todia()

        
        #
        # Initialize vectors
        #
        cls.v = np.zeros_like( cls.u )
        cls.w = np.zeros_like( cls.u )
        cls.b = np.zeros_like( cls.u )
        cls.b1 = np.zeros_like( cls.u )
        cls.b2 = np.zeros_like( cls.u )

        #
        # Time stepping
        #
        for i in range( len( cls.t )-1 ):

            # Calculate v vector
            cls.v[i,...] = cls.u[i,...]

            # Calculate w vector
            cls.w[i,...] = ( cls.u[i,...] ** 2 )/2

            # Calculate b vector
            cls.b1[i,...] = cls.C_matrix.dot( cls.v[i,...] ) 
            cls.b2[i,...] = cls.D_matrix.dot( cls.w[i,...] )
            cls.b[i,...] = cls.b1[i,...] + cls.b2[i,...]

            # Solve u = A\b
            cls.u[i+1,:] = spsr.linalg.spsolve( cls.A_matrix , cls.b[i,...] )

    def loss(cls, loss_norm=2.0, engine="numpy" ):
        """
            This method calculates the loss function of the Burger's equation results.

        Args:
            loss_norm (float, optional):    The norm to calculate the loss function by. Defaults to
                                                2.0.

            engine (string, optional):  Which engine will find the losses of the solution. The 
                                            valid options are:

                                        - *"numpy","np","mkl":  Numpy

                                        - "torch","pytorch":    Pytorch

                                        The default is "numpy". Not case sensitive.

        """

        if engine.lower() in ["numpy", "np", "mkl"]:
            
            # Import numpy as our unkown engine
            xp = np

            ( cls.du_dt, cls.du_dx ) = xp.gradient( cls.u, cls.dt, cls.dx, edge_order=2 )
            cls.d2u_dx2 = xp.gradient( cls.du_dx, cls.dx, axis=-1, edge_order=2 )

            points = xp.prod( xp.shape( cls.losses ) )

        elif engine.lower() in ["torch","pytorch"]:
            
            # Import pytorch and make our unkown engine
            import torch
            xp = torch

            u = torch.tensor( cls.u , require_grad=True )
            dt = torch.tensor( cls.dt )
            dx = torch.tensor( cls.dx )

            # Compute gradients along different dimensions
            cls.du_dt = torch.autograd.grad(outputs=u.sum(), inputs=u, create_graph=True)[0]/dt
            cls.du_dx = torch.autograd.grad(outputs=u.sum(dim=0), inputs=u, create_graph=True)[0]/dx
            cls.d2u_dx2 = torch.autograd.grad(outputs=du_dx.sum(dim=0), inputs=u, create_graph=True)[0]/dx

            # Compute the residual losses of the equation
            points = torch.prod(torch.tensor(cls.du_dt.shape, dtype=torch.float32))

        cls.losses = ( cls.du_dt - ( cls.nu*cls.d2u_dx2 - cls.u*cls.du_dx ) ) ** loss_norm
        cls.loss_total = xp.sum( cls.losses ) / points
    
#==================================================================================================
#
# Kuramoto-Sivashinsky Equation Objects
#
#==================================================================================================

class KS_og:
    """
    This object contains the necessary data and functions to solve the
        Kuramoto-Sivashinsky equations

    """

    def __init__( self , x , u_0 , t_bounds , dt , alpha=-1e-6 , beta=0.0 , gamma=1e-6 ):
        """
        This method initialized the KS equation object to set up the solver.

        Args:
            x [float]:      The spatial domain to calculate the KS equation
                                over.

            u_0 [float]:    The initial values of the function to initialize
                                the KS equation.

            t_bounds (float):   The bounds of time to solve over.

            dt (float):     The time step size for the solution.

            alpha (float, optional):  The value for the \alpha coefficient. 
                                            Defaults to 1.0.

            beta (float, optional): The value for the \beta coefficient. 
                                            Defaults to 0.0.

            gamma (float, optional):    The value for the \gamma coefficient. 
                                            Defaults to 1.0.

        Attributes:
            x   <-  x

            u_0 <-  u_0

            t_st (float):   The starting time for the KS solve. min(t_bounds)

            t_sp (float):   The end time for the KS solve. max(t_bounds)

            dt  <-  dt

            alpha   <-  alpha
            
            beta    <-  beta

            gamma   <-  gamma

        """

        self.x = x
        self.u_0 = u_0
        self.dt = dt
        self.t_st = np.min( t_bounds )
        self.t_sp = np.max( t_bounds )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.t = np.arange( self.t_st , self.t_sp , self.dt )
        self.dx = np.mean( np.gradient( self.x ) )

        self.Nx = len( self.x )
        self.Nt = len( self.t )

        self.u = np.zeros(  ( self.Nt , self.Nx ) )
        self.u[0,:] = u_0
        self.v = np.zeros_like( self.u )
        self.f = np.zeros_like( self.u )

        self.Re_cell = np.max( np.abs( self.u ) ) * self.dx / -self.alpha
        self.c_cell = np.sqrt( self.gamma *  np.max( np.abs( self.u ) ) / ( self.dx ** 3 ) )
        self.Ma_cell = np.max( np.abs( self.u ) ) / self.c_cell

    def solve( cls , n_xOrder=4 , n_tOrder=4 , bc_u=(0,0) , bc_dudx=(0,0) , bc_d2udx2=(None,None) , bc_d3udx3=(None,None) , bc_d4udx4=(None,None) , bc_xOrder=1 , zero_tol=1e-12 ):
        """
        Solve the KS equation as initialized.

        The solver equation takes the form:

        D<u_k+1>=A<v_k>+B<u_k>+E<e>

            where v = (u^2/2) and E<e> represents the boundary condition 
                solution

        Args:
            n_xOrder (int, optional): The spatial order of accuracy. 
                                        Defaults to 4.
            n_tOrder (int, optional): The time order of accuracy. The input
                                        values correspond to:
                                        
                                    - 1: Euler time stepping

                                    - 2:    NOPE

                                    - 3:    NOPE

                                    - 4: Runge-Kutta-4 time stepping
                                        
                                        Defaults to 4.

        """
        #
        # Set up time stepping parameters
        #
        cls.f = np.zeros_like( cls.u )
        cls.phi = np.zeros_like( cls.u )
        if n_tOrder==1:
            print("Eulerian time stepping selected")
        elif n_tOrder==4:
            print("RK4 time stepping selected.")
            cls.R_n = np.zeros_like( cls.u )
            cls.R_1 = np.zeros_like( cls.u )
            cls.R_2 = np.zeros_like( cls.u )
            cls.R_3 = np.zeros_like( cls.u )
            cls.u_1 = np.zeros_like( cls.u )
            cls.u_2 = np.zeros_like( cls.u )
            cls.u_3 = np.zeros_like( cls.u )
            cls.v_1 = np.zeros_like( cls.u )
            cls.v_2 = np.zeros_like( cls.u )
            cls.v_3 = np.zeros_like( cls.u )
            cls.f_R = np.zeros_like( cls.u )
            cls.phi_1 = np.zeros_like( cls.u )
            cls.phi_2 = np.zeros_like( cls.u )
            cls.phi_3 = np.zeros_like( cls.u )

        #
        # Calculate the matrix for advective term
        #
        cls.numgradient_advect = numericalGradient( 1 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_advect.formMatrix( cls.Nx )
        cls.A_advect = cls.numgradient_advect.gradientMatrix / cls.dx
        # Change boundary rows
        cls.A_advect = cls.A_advect.tolil()
        cls.numgradient_LHS_advect = numericalGradient( 1 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.A_advect[i,:]=0
            cls.A_advect[i,i:i+n_xOrder+1] = cls.numgradient_LHS_advect.coeffs / cls.dx
        cls.numgradient_RHS_advect = numericalGradient( 1 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.A_advect[-i-1,:]=0
            if i>0:
                cls.A_advect[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_advect.coeffs / cls.dx
            else:
                cls.A_advect[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_advect.coeffs / cls.dx
        # Fix float zeros
        cls.A_advect[np.abs(cls.A_advect)*cls.dx<=zero_tol]=0
        cls.A_advect = cls.A_advect.todia()

        #
        # Calculate the matrix for diffusive term
        #
        cls.numgradient_diffuse = numericalGradient( 2 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_diffuse.formMatrix( cls.Nx )
        cls.B_diffuse = cls.numgradient_diffuse.gradientMatrix / ( cls.dx ** 2)
        # Change boundary rows
        cls.B_diffuse = cls.B_diffuse.tolil()
        cls.numgradient_LHS_diffuse = numericalGradient( 2 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_diffuse[i,:]=0
            cls.B_diffuse[i,i:i+n_xOrder+1] = cls.numgradient_LHS_diffuse.coeffs / ( cls.dx ** 2)
        cls.numgradient_RHS_diffuse = numericalGradient( 2 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_diffuse[-i-1,:]=0
            if i>0:
                cls.B_diffuse[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_diffuse.coeffs / ( cls.dx ** 2)
            else:
                cls.B_diffuse[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_diffuse.coeffs / ( cls.dx ** 2)
        cls.B_diffuse = cls.B_diffuse.todia()  

        #
        # Calculate the matrix for the 3rd derivative term
        #
        cls.numgradient_third = numericalGradient( 3 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_third.formMatrix( cls.Nx )
        cls.B_third = cls.numgradient_third.gradientMatrix / ( cls.dx ** 3 )
        # Change boundary rows
        cls.B_third = cls.B_third.tolil()
        cls.numgradient_LHS_third = numericalGradient( 3 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_third[i,:]=0
            cls.B_third[i,i:i+n_xOrder+1] = cls.numgradient_LHS_third.coeffs / ( cls.dx ** 3 )
        cls.numgradient_RHS_third = numericalGradient( 3 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_third[-i-1,:]=0
            if i>0:
                cls.B_third[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_third.coeffs / ( cls.dx ** 3 )
            else:
                cls.B_third[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_third.coeffs / ( cls.dx ** 3 )
        cls.B_third = cls.B_third.todia() 

        #
        # Calculate the matrix for the 4th derivative term
        #
        cls.numgradient_fourth = numericalGradient( 4 , ( n_xOrder - n_xOrder//2 , n_xOrder//2 ) )
        cls.numgradient_fourth.formMatrix( cls.Nx )
        cls.B_fourth = cls.numgradient_fourth.gradientMatrix / ( cls.dx ** 4 )
        # Change boundary rows
        cls.B_fourth = cls.B_fourth.tolil()
        cls.numgradient_LHS_fourth = numericalGradient( 4 , ( 0 , n_xOrder ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_fourth[i,:]=0
            cls.B_fourth[i,i:i+n_xOrder+1] = cls.numgradient_LHS_fourth.coeffs / ( cls.dx ** 4 )
        cls.numgradient_RHS_fourth = numericalGradient( 4 , ( n_xOrder , 0 ) )
        for i in range( int(np.rint(n_xOrder/2)) ):
            cls.B_fourth[-i-1,:]=0
            if i>0:
                cls.B_fourth[-i-1,-(i+n_xOrder+1):-i] = cls.numgradient_RHS_fourth.coeffs / ( cls.dx ** 4 )
            else:
                cls.B_fourth[-i-1,-(n_xOrder+1):] = cls.numgradient_RHS_fourth.coeffs / ( cls.dx ** 4 )
        cls.B_fourth = cls.B_fourth.todia()

        #
        # Combine matrices
        #  
        cls.A = -cls.A_advect
        cls.B = -cls.alpha * cls.B_diffuse - cls.beta * cls.B_third - cls.gamma * cls.B_fourth

        #
        # Create matrix for the LHS
        #
        cls.D = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )

        #
        # Create matrix for the Neumann BC's
        #
        cls.E = spsr.dia_matrix( ( np.ones( cls.Nx ) , [0] ) , shape = ( cls.Nx , cls.Nx ) )
        cls.e = np.zeros( cls.Nx )

        #
        # Place boundary conditions
        #
        bc_count=0
        bc_LHS_count=0
        bc_RHS_count=0
        cls.A = cls.A.tolil()
        cls.B = cls.B.tolil()
        cls.D = cls.D.tolil()
        cls.E = cls.E.tolil()
        # Neumann boundary condition
        if bc_u[0]:
            cls.e[0] = bc_u[0]
        if bc_u[-1]:
            cls.e[-1] = bc_u[-1]
        for i , bc in enumerate( bc_u ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    cls.A[0,:] = 0
                    cls.B[0,:] = 0
                    cls.D[0,:] = 0
                    cls.D[0,0] = 1
                    cls.E[0,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    cls.A[-1,:] = 0
                    cls.B[-1,:] = 0
                    cls.D[-1,:] = 0
                    cls.D[-1,-1] = 1
                    cls.E[-1,-1] = 1
                    bc_RHS_count += 1
        # Dirichlet boundary condition
        if bc_dudx[0]:
            cls.e[0] = bc_dudx[0]
        if bc_dudx[-1]:
            cls.e[-1] = bc_dudx[-1]
        for i , bc in enumerate( bc_dudx ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 1 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 1 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        # Diffusion boundary condition
        if bc_d2udx2[0]:
            cls.e[bc_LHS_count] = bc_d2udx2[0]
        if bc_d2udx2[-1]:
            cls.e[-1-bc_RHS_count] = bc_d2udx2[-1]
        for i , bc in enumerate( bc_d2udx2 ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 2 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 2 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        # Third order derivative boundary condition
        if bc_d3udx3[0]:
            cls.e[bc_LHS_count] = bc_d3udx3[0]
        if bc_d3udx3[-1]:
            cls.e[-1-bc_RHS_count] = bc_d3udx3[-1]
        for i , bc in enumerate( bc_d3udx3 ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 3 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 3 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        # Fourth order derivative boundary condition
        if bc_d4udx4[0]:
            cls.e[bc_LHS_count] = bc_d4udx4[0]
        if bc_d4udx4[-1]:
            cls.e[-1-bc_RHS_count] = bc_d4udx4[-1]
        for i , bc in enumerate( bc_d4udx4 ):
            if bc or bc==0:
                bc_count += 1

                if i==0:
                    numgradient_BC = numericalGradient( 4 , ( 0 , bc_xOrder ) )
                    cls.A[bc_LHS_count,:] = 0
                    cls.B[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:] = 0
                    cls.D[bc_LHS_count,:bc_xOrder+1] = numgradient_BC.coeffs
                    cls.E[bc_LHS_count,0] = 1
                    bc_LHS_count += 1

                if i==len(bc_u)-1:
                    numgradient_BC = numericalGradient( 4 , ( bc_xOrder , 0 ) )
                    cls.A[-1-bc_RHS_count,:] = 0
                    cls.B[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,:] = 0
                    cls.D[-1-bc_RHS_count,-(bc_xOrder+1):] = numgradient_BC.coeffs
                    cls.E[-1-bc_RHS_count,-1] = 1
                    bc_RHS_count += 1
        if bc_count>4:
            raise ValueError( "Too many boundary conditions present, only 4x are allowed." )
        elif bc_count<4:
            raise ValueError( "Too few boundary conditions present, 4x are required.")
        cls.A.tocsr()
        cls.B.tocsr()
        cls.D.tocsr()
        cls.E.tocsr()
        cls.Ee = cls.E.dot( cls.e )
        #cls.Ee[np.abs(cls.Ee)<=zero_tol]=0
            
        #
        # Time stepping
        #
        for i in range( cls.Nt-1 ):
            
            #
            # Set up RHS
            #
            cls.v[i,:] = (cls.u[i,:]**2)/2
            cls.Av_k = cls.A.dot( cls.v[i,:] )
            cls.Bu_k = cls.B.dot( cls.u[i,:] )
            cls.f[i,:] = cls.Av_k + cls.Bu_k + cls.Ee

            #
            # Time step
            #
            if i<=n_tOrder or n_tOrder==1:

                cls.phi[i,:] = cls.f[i,:] * cls.dt + cls.u[i,:]
                cls.phi[i,np.abs(cls.phi[i,:])<=zero_tol]=0
                cls.phi[i,-2:]=cls.Ee[-2:]
                cls.phi[i,:2]=cls.Ee[:2]
                cls.u[i+1,:] = spsr.linalg.spsolve( cls.D , cls.phi[i,:] )

            elif n_tOrder==4:
                cls.R_n[i,:] = cls.f[i,:]

                #
                # Perform virtual step (1)
                #
                cls.phi_1[i,:] = (cls.dt/2)*cls.R_n[i,:] + cls.u[i,:]
                #cls.phi_1[i,:] = (cls.dt/2)*spsr.linalg.spsolve( cls.D , cls.R_n[i,:] ) + cls.u[i,:]
                #cls.u_1[i,:] = cls.phi_1[i,:]
                cls.phi_1[i,np.abs(cls.phi_1[i,:])<=zero_tol]=0
                cls.phi_1[i,-2:]=cls.Ee[-2:]
                cls.phi_1[i,:2]=cls.Ee[:2]
                cls.u_1[i,:] = spsr.linalg.spsolve( cls.D , cls.phi_1[i,:] )
                cls.v_1[i,:] = (cls.u_1[i,:] ** 2)/2
                cls.R_1[i,:] = cls.A.dot( cls.v_1[i,:] ) + cls.B.dot( cls.u_1[i,:] ) + cls.Ee

                #
                # Perform virtual step (2)
                #
                cls.phi_2[i,:] = (cls.dt/2)*cls.R_1[i,:] + cls.u[i,:]
                #cls.phi_2[i,:] = (cls.dt/2)*spsr.linalg.spsolve( cls.D , cls.R_1[i,:] ) + cls.u[i,:]
                #cls.u_2[i,:] = cls.phi_2[i,:]
                cls.phi_2[i,np.abs(cls.phi_2[i,:])<=zero_tol]=0
                cls.phi_2[i,-2:]=cls.Ee[-2:]
                cls.phi_2[i,:2]=cls.Ee[:2]
                cls.u_2[i,:] = spsr.linalg.spsolve( cls.D , cls.phi_2[i,:] )
                cls.v_2[i,:] = (cls.u_2[i,:] ** 2)/2
                cls.R_2[i,:] = cls.A.dot( cls.v_2[i,:] ) + cls.B.dot( cls.u_2[i,:] ) + cls.Ee

                #
                # Perform virtual step (3)
                #
                cls.phi_3[i,:] = cls.dt*cls.R_2[i,:] + cls.u[i,:]
                #cls.phi_3[i,:] = cls.dt*spsr.linalg.spsolve( cls.D , cls.R_2[i,:] ) + cls.u[i,:]
                #cls.u_3[i,:] = cls.phi_3[i,:]
                cls.phi_3[i,np.abs(cls.phi_3[i,:])<=zero_tol]=0
                cls.phi_3[i,-2:]=cls.Ee[-2:]
                cls.phi_3[i,:2]=cls.Ee[:2]
                cls.u_3[i,:] = spsr.linalg.spsolve( cls.D , cls.phi_3[i,:] )
                cls.v_3[i,:] = (cls.u_3[i,:] ** 2)/2
                cls.R_3[i,:] = cls.A.dot( cls.v_3[i,:] ) + cls.B.dot( cls.u_3[i,:] ) + cls.Ee

                #
                # Finish the time step
                #
                cls.phi[i,:] = cls.u[i,:] + (cls.dt/6)*( cls.R_n[i,:] + 2*cls.R_1[i,:] + 2*cls.R_2[i,:] + cls.R_3[i,:] )
                cls.phi[i,np.abs(cls.phi[i,:])<=zero_tol]=0
                cls.phi[i,-2:]=cls.Ee[-2:]
                cls.phi[i,:2]=cls.Ee[:2]
                cls.u[i+1,:] = spsr.linalg.spsolve( cls.D , cls.phi[i,:] )

                