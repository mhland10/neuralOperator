"""

**burgers_datageneration.py**

@Author:    Matthew Holland
@Date:      2025-02-18
@Version:   0.0
@Contact:   matthew.holland@my.utsa.edu

    This file contains the data generation objects for the Burgers equation. It will take the input
of constants for a tanh function and some noise and generate a bunch of data to train the neural
network on via storage in an h5 file.

"""

#==================================================================================================
#
#   Importing Required Modules
#
#==================================================================================================

import os, sys
# Get the absolute path of the ../solvers directory
solvers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../solvers'))
print(f"Solvers path:\t{solvers_path}")
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../solvers/lib'))
sys.path.append(solvers_path)

import numpy as np
import h5py as h5
from solvers import *
import matplotlib.pyplot as plt
from distributedFunctions import *
#from mpi4py import COMM_WORLD

#==================================================================================================
#
#   Set up calculation parameters
#
#==================================================================================================

# Set up initial courant number
C = 0.1

# Set up initial spatial domain
x = np.linspace( -2*np.pi , 2*np.pi , num=200 )

# Set the end point
t_end = 0.5

#
# Parameters to vary
#
As = np.logspace( -2 , 0 , num=5 )
cs = np.logspace( -2 , 0 , num=5 )
snrs = np.logspace( -3 , -1 , num=5 )
cc, AA, ss = np.meshgrid( cs , As, snrs )
print(f"Meshgrid shape:\t{np.shape(AA)}")

# Indices to go along with parameters
i_s = np.arange( 0, len(As) )
j_s = np.arange( 0, len(cs) )
k_s = np.arange( 0, len(snrs) )
ii, jj, kk = np.meshgrid( i_s, j_s, k_s )
i_flat = ii.flatten()
j_flat = jj.flatten()
k_flat = kk.flatten()

# Initial conditions
u_0_s = np.zeros( np.shape( AA ) + (len(x) ,) )

# Spatial order for solution
space_order = 4

#
# Move to script directory
#
#os.chdir(os.path.realpath(__file__)[:-34])


#==================================================================================================
#
#   Calculate the initial conditions to the Burgers equation
#
#==================================================================================================

# Here, we sweep through the various parameters to create a set of initial conditions
for i in range( len( i_flat ) ):
    # Define the matrix location for the meshgrid from the flatten indices
    loc = (i_flat[i],j_flat[i],k_flat[i])
    print(f"Location:\t{loc}")
    # Establish the initial conditions via the u(0, x)=-A * ( c sin(x) - 2 sin(x/2) ) * (1+noise)
    u_0_s[loc] = AA[loc] * ( cc[loc] * np.sin( x ) - 2 * np.sin( x/2 ) ) * ( 1 + ss[loc] * np.random.rand( len(x) )/2 - ss[loc] * np.random.rand( len(x) )/2 )


#==================================================================================================
#
#   Plot the initial conditions to the Burgers equation
#
#==================================================================================================

# Here, we can plot the initial conditions via the SNR to see what we are producing, or one can 
#   comment them out

for k in k_s:
    for i in i_s:
        for j in j_s:
            plt.plot( x , u_0_s[i,j,k], label="A={x:.3e}, c={y:.3e}".format(x=As[i], y=cs[j]))
    plt.title("Initial Conditions for SNR={x:.3e}".format(x=snrs[k]))
    plt.xlabel("x Position [m]")
    plt.ylabel("u\u2080 [m/s]")
    plt.legend(loc="best")
    plt.show()
#"""

#==================================================================================================
#
#   Calculate the Burgers equation
#
#==================================================================================================

# Perform the burgers equation functions
burgers = []
for i in range( len( i_flat ) ):
    # Define our location in the sweep
    loc = (i_flat[i],j_flat[i],k_flat[i])
    # Initialize the burgers equation object - in this one we will use the original Burger's 
    #   equation that came out of the CFD class
    print(f"Solving Burger index {i}: A={AA[loc]:.3e}, c={cc[loc]:.3e}, snr={ss[loc]:.3e}")
    hello = burgers1D( x, u_0_s[loc], (0, t_end), dt=10.0e-6, spatial_order=space_order, nu=0.01 )
    #hello = burgersEquation_og(x, u_0_s[loc], (0, t_end), C=C )
    # Solve the Burger's equation via a spatial order of 2, this can be changed
    hello.solve()
    # Calculate the loss to see the error from our Burger's equation solve
    #hello.loss()
    burgers += [hello]

#==================================================================================================
#
#   Write to data
#
#==================================================================================================

# Write all our data by case to a *.h5 file
# I split the defining data for the dataset into attributes for easier access
with h5.File("dualSine_data.h5", "w") as f:
    print(f"Writing file in {os.getcwd()}")
    # Create group
    for i in range( len( i_flat ) ):
        loc = (i_flat[i],j_flat[i],k_flat[i])
        group = f.create_group(f"dataset-{i}")

        group.create_dataset("u_0", data=u_0_s[loc])
        group.attrs["A"]=AA[loc]
        group.attrs["c"]=cc[loc]
        group.attrs["snr"]=ss[loc]
        group.create_dataset("u", data=burgers[i].u)
        #group.create_dataset("losses", data=burgers[i].losses)
        #group.attrs["total_losses"]=burgers[i].loss_total

#==================================================================================================
#
#   Plot the data
#
#==================================================================================================   

# Plot the data to see what we are producing, or one can
#   comment them out

for i in range( len( i_flat ) ):
    b = burgers[i]


    plt.contourf( b.x, b.t, b.u )
    plt.xlabel("x Position [m]")
    plt.ylabel("Time [s]")
    plt.title("Burgers Equation Solution")
    plt.colorbar(label="u [m/s]")
    plt.show()
#***

