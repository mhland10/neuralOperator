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
from solvers import burgersEquation_og
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
x = np.linspace( -1 , 1 , num=200 )

# Set the end point
t_end = 1.0

#
# Parameters to vary
#
As = np.logspace( -2 , 0 , num=5 )
cs = np.logspace( 0 , 2 , num=5 )
snrs = np.logspace( -3 , -1 , num=5 )
AA, cc, ss = np.meshgrid( As , cs, snrs )

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

#
# Move to script directory
#
os.chdir(os.path.realpath(__file__)[:-30])

#==================================================================================================
#
#   Calculate the initial conditions to the Burgers equation
#
#==================================================================================================

for i in range( len( i_flat ) ):
    loc = (i_flat[i],j_flat[i],k_flat[i])
    u_0_s[loc] = -AA[loc] * np.tanh( cc[loc] * x ) * ( 1 + ss[loc] * np.random.rand( len(x) )/2 - ss[loc] * np.random.rand( len(x) )/2 )


#==================================================================================================
#
#   Plot the initial conditions to the Burgers equation
#
#==================================================================================================

"""
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

burgers = []
for i in range( len( i_flat ) ):
    loc = (i_flat[i],j_flat[i],k_flat[i])
    hello = burgersEquation_og(x, u_0_s[loc], (0, t_end), C=C )
    hello.solve( N_spatialorder=2 )
    hello.loss()
    burgers += [hello]

#==================================================================================================
#
#   Write to data
#
#==================================================================================================

with h5.File("tanh_data.h5", "w") as f:
    # Create group
    for i in range( len( i_flat ) ):
        loc = (i_flat[i],j_flat[i],k_flat[i])
        group = f.create_group(f"dataset-{i}")

        group.create_dataset("u_0", data=u_0_s[loc])
        group.create_dataset("A", data=AA[loc])
        group.create_dataset("c", data=cc[loc])
        group.create_dataset("snr", data=ss[loc])
        group.create_dataset("u", data=burgers[i].u)
        group.create_dataset("losses", data=burgers[i].losses)
        group.create_dataset("total_losses", data=burgers[i].loss_total)

