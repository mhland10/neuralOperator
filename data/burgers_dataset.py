import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import sys
import torch
from torch import nn
from scipy.integrate import odeint


##########################################
############## EQUATION SOLVING ##########
##########################################

#Definition of ODE system (PDE ---(FFT)---> ODE system)
def burg_system(u, t, k, mu, nu):
    #Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j*k*u_hat
    u_hat_xx = -k**2*u_hat
    
    #Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)
    
    #ODE resolution
    u_t = -mu*u*u_x + nu*u_xx
    return u_t.real


##########################################
############## Dataset Object   ##########
##########################################    
class Dataset(torch.utils.data.Dataset):
  def __init__(self, cnt=1000, test=False):
    self.cnt = cnt
    self.test = test
    ############## SET-UP THE PROBLEM ###############
    self.mu = 1
    self.nu = 0.01       #kinematic viscosity coefficient
        
    #Spatial mesh
    L_x = 10                   #Range of the domain according to x [m]
    self.dx = 0.01                  #Infinitesimal distance
    N_x = int(L_x/self.dx)          #Points number of the spatial mesh
    self.X = np.linspace(0,L_x,N_x) #Spatial array

    #Temporal mesh
    L_t = 1                     #Duration of simulation [s]
    self.dt = 0.025                  #Infinitesimal time
    N_t = int(L_t/self.dt)           #Points number of the temporal mesh
    self.T = np.linspace(0,L_t,N_t)  #Temporal array
    self.N_t = self.T.shape[0]

    #Wave number discretization
    self.k = 2*np.pi*np.fft.fftfreq(N_x, d = self.dx)


  def domain_change(self, N_x, N_t):
     # space
     L_x = 10
     self.dx = L_x / N_x
     self.X = np.linspace(0, L_x, N_x)

     # time
     L_t = 1
     self.dt = L_t / N_t
     self.T = np.linspace(0, L_t, N_t)
     self.N_t = self.T.shape[0]

     #Wave number discretization
     self.k = 2*np.pi*np.fft.fftfreq(N_x, d = self.dx)
    

  def __len__(self):
      return self.cnt


  def __getitem__(self, idx):
     # Seed if test
     if self.test: 
        np.random.seed(idx)

     # Pick an initial condition
     cx = np.random.uniform(2,10-2)
     var = np.random.uniform(.25,3) # original is 2
     u0 = np.exp(-(self.X-cx)**2/var)
     
     # Solve over all time
     U = odeint(burg_system, u0, self.T, args=(self.k, self.mu, self.nu,), mxstep=5000).T
     
     # Convert to torch
     u0 = torch.from_numpy(u0.astype(np.float32) ) # N_x
     U = torch.from_numpy( U.astype(np.float32)  ) # N_x x N_t
     
     u0 = u0[None,:]
     U = U[None,:,:]

     return u0, U
     


