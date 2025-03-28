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
                if np.max(Cos)>cls.C:
                    cls.adaptiveTimeStepping(x_domain, cls.dt[-1])
                else:
                    cls.timeStep(x_domain)
                
            else:
                if np.max(Cos)>1.0:
                    cls.adaptiveTimeStepping(x_domain, cls.dt[-1])
                else:
                    cls.timeStep(x_domain)

        #
        # Reset to numpy arrays
        #
        cls.t = np.array(cls.t)
        try:
            cls.u = np.array(cls.u)
        except:
            raise Warning("Could not convert u to numpy array")
        
    def timeStep(cls, x_domain):
        """
            This method performs the actual time stepping.

        Args:
            x_domain (numpy ndarray - float): The domain of the problem.

        """

        # Find the time derivative
        du_dt = cls.eqn( x_domain, cls.u[-1], cls.coeffs, cls.BCs )

        # Perform the integration
        cls.u += [cls.u[-1] + cls.dt[-1]*du_dt]

        # Add time steps
        cls.t += [cls.dt[-1]+cls.t[-1]]
        

    def adaptiveTimeStepping(cls, x_domain, dt ):
        """
            In this method, the object will perform the adaptive time stepping to hold the Courant
        Number under the specified value. This is done by checking the Courant number and then
        calculating the new time step size based on the maximum Courant number and current velocity
        and x-domain.

        Args:
"
            x_domain (numpy ndarray - float): The domain of the problem.

            dt (float): The original time step size.

        """
        print("**Using adaptive Time Stepping**")

        # Get the maximum allowable time step
        dt_new = cls.C / np.max(np.abs(cls.u[-1])/np.gradient(x_domain))
        print(f"New time step: {dt_new}")

        # Set a new array of time steps
        t_s_new = list( cls.t[-1] + np.arange( 0, dt, dt_new) ) + [cls.t[-1]+dt]
        t_s_new = np.array(t_s_new)
        print(f"New time steps: {t_s_new}")

        # Solver over the new time steps
        for i, tt in enumerate(t_s_new):
            print(f"Time: {tt}")
            # Find the time derivative
            du_dt = cls.eqn( x_domain, cls.u[-1], cls.coeffs, cls.BCs )

            # Perform the integration
            dt_step = tt - cls.t[-1]
            print(f"t_step: {dt_step}")
            cls.u += [cls.u[-1] + dt_step*du_dt]

            # Create a new Courant number
            Cos_new = np.abs(cls.u[-1])*dt_step/np.gradient(x_domain)
            print(f"Maximum velocity:\t{np.max(np.abs(cls.u[-1]))}")
            print(f"Minimum step size:\t{np.min(np.gradient(x_domain))}")
            print(f"New Maximum Courant Number: {np.max(Cos_new)}")

            # Add time steps
            cls.t += [dt_step+cls.t[-1]]



