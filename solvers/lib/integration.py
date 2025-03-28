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
from equation import *

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
    def __init__(self, eqn, x, u_0, t_domain, dt, coeffs, BCs, C=None, spatial_order=None, spatialBC_order=None, initialized=True ):
        """
            What happens when the explicit Euler object is called.

        Args:
            eqn (string):   The equation object that will be integrated via the Euler stepping
                                method. Must be initialized by the time that the integrator accepts
                                it.

            x (numpy ndarray - float):  The domain, or at least initial domain.

            u_0 (numpy ndarray - float):    The initial values of the function, u, in the domain.

            t_domain (float):   The times that the solve will be done over. Must be in the format:

                                (t_start, t_end)

            dt (float): The (initial) time step for the time stepping.
            
            coeffs (tuple - float): The tuple of coefficients to pass to the equation object. Must
                                        correspond to the coefficients that the equation needs.

            BCs (list): The list of the boundary conditions. Must be tuples of (2) in order of 
                            least order gradient to higher grdients. Can be a value if a value is 
                            the boundary condition, or "same" if the boundary just needs to hold a
                            constant value.

            C (float, optional):    The maximum allowable Courant number for the solver. Leave as
                                        None to allow dt time stepping only. Will split the time 
                                        step if the stepping exceeds the Courant number. Defaults
                                        to None.

        """
        # Set up the inputs to put into the equation object
        self.x_0 = x
        self.t_start = t_domain[0]
        self.t_end = t_domain[1]
        self.C = C
        self.BCs = BCs
        self.u_0 = u_0
        self.u = [self.u_0]
        self.t = [self.t_start]
        self.x = [self.x_0]
        self.dx=[]
        self.dt = [dt]
        self.coeffs = coeffs


        # Initialize the equation
        self.eqn = eqn
        
        # Set things
        self.dyn_mesh = False #TODO: At some point, we should add a dynamic mesh
        

    def solve(cls, time_deriv_store=True ):
        """
            Solve the Euler stepping 

        """

        # Initialize the time derivative storage
        if time_deriv_store:
            cls.time_deriv = []

        # Define the time steps to use
        t_s = list(np.arange(cls.t_start, cls.t_end, cls.dt[-1]))
        if cls.t_end not in t_s:
            t_s += [cls.t_end]
        t_s = np.array(t_s)
        cls.t_s = t_s

        # Loop over the time steps
        for i, t in enumerate(t_s):
            print(f"Time: {cls.t[-1]}")

            # Define the domain
            if cls.dyn_mesh:
                cls.dx+=[np.gradient(cls.x[0])]
                dx_step = cls.dx[0]
            x_domain = cls.x[-1]

            # Check the Courant number
            Cos = np.abs(cls.u[-1])*cls.dt[-1]/np.gradient(x_domain)
            print(f"\tMaximum Courant Number: {np.max(Cos)}")
            if cls.C:
                print("Checking Courant Number")
            else:
                if np.max(Cos)>1.0:
                    print("No Courant Number Check, defaulting to Co=1.0")
                    dt_reduced = 1/np.max( np.abs(cls.u[-1])/np.gradient(x_domain) )
                    print(f"The reduced dt is: {dt_reduced:.2e} from {cls.dt[-1]:.2e}")
                    t_s_new = np.arange(cls.t[-1]+dt_reduced, cls.t[-1]+cls.dt[-1], dt_reduced)
                    print(f"New time steps are: {t_s_new}")

            # Find the time derivative
            du_dt = cls.eqn( x_domain, cls.u[-1], cls.coeffs, cls.BCs )

            # Perform the integration
            cls.u += [cls.u[-1] + cls.dt[-1]*du_dt]

            

            cls.t += [cls.dt[-1]+cls.t[-1]]

        #
        # Reset to numpy arrays
        #
        cls.t = np.array(cls.t)
        try:
            cls.u = np.array(cls.u)
        except:
            raise Warning("Could not convert u to numpy array")


