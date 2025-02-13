from collections import OrderedDict
import torch
import torch.nn as nn

# Define the scale-aware model elements
class scale_aware_conv2d(nn.Module):
  def __init__(self, in_channels=1, out_channels=1, ksize=3, bias=True):
      super(scale_aware_conv2d, self).__init__()
      self.W = torch.nn.Parameter( 1e-4*torch.randn((out_channels, in_channels, ksize, ksize) ) )
      if bias:
        self.b = torch.nn.Parameter( 1e-6*torch.randn((out_channels,)) )
      else:
        self.b = None
    
  def forward(self, x, dx):
    xn = nn.functional.conv2d(x, (1/dx) * self.W, padding=1)
    if self.b is not None: xn += self.b[None,:,None,None]
    return xn

##############################
## Scale-aware 1D Conv     ###
##############################
class scale_aware_conv1d_alt(nn.Module):
  def __init__(self, in_channels=1, out_channels=1, ksize=3, bias=True, ps=[0,-1]):
    super(scale_aware_conv1d_alt, self).__init__()
    self.ps = ps
    self.convs = torch.nn.ModuleDict()    # Dictionary- a conv for every power of dx
    for p in self.ps:
      # Circular padding mode to mimic circular boundary conditions, bias is not part of convs
      self.convs[str(p)] = torch.nn.Conv1d(in_channels, out_channels, ksize, padding=1, padding_mode='circular', bias=False)
    if bias:
      # bias as a separate learnable parameter
      self.b = torch.nn.Parameter( 1e-4 * torch.randn((out_channels,)) )
    else:
      self.b = None

  def forward(self, x, dx):
    ys = []
    for p in self.ps:
      ys.append( self.convs[str(p)](x) * dx**p )
    y = torch.sum(torch.stack(ys, dim=1),dim=1)
    if self.b is not None:
      y = y + self.b[None,:,None]
    return y

##############################
## Scale-aware 1D Conv     ###
##############################
# My first implementation, and we're not using it
class scale_aware_conv1d(nn.Module):
  def __init__(self, in_channels=1, out_channels=1, ksize=3, bias=True, ps=[0,-1] ):
      super(scale_aware_conv1d, self).__init__()
      self.ps = ps
      self.W = torch.nn.Parameter( 1e-4 * torch.randn((len(ps), out_channels, in_channels, ksize) ) )
      if bias:
        self.b = torch.nn.Parameter( 1e-4 * torch.randn((out_channels,)) )
      else:
        self.b = None
    
  def forward(self, x, dx):
    xs = []
    for i, p in enumerate( self.ps ):
      xs.append( nn.functional.conv1d(x, dx**p * self.W[i], padding=1) )
    xn = torch.sum(torch.stack(xs),dim=0)

    if self.b is not None: xn += self.b[:,None]
    return xn
    
  # Each of these layers could have sets of weights / split computations for dx^-2 through dx^2
    
class Upsample1D(nn.Module):
  def __init__(self):
    super(Upsample1D, self).__init__()
    self.resample = nn.Upsample(scale_factor=2, mode='nearest')  

  def forward(self, x):
    xn = self.resample(x[:,None,:])[:,0,:]
    return xn

##############################
## Conv1d multiscale model ###
##############################

class block(nn.Module):
  def __init__(self, in_channels, out_channels, mode=None, ps=[0,-1]):
    super(block, self).__init__()
    self.a = scale_aware_conv1d_alt(in_channels, out_channels, ps=ps)
    self.b = scale_aware_conv1d_alt(out_channels, out_channels, ps=ps)
    self.r = torch.nn.ELU()
    
    if mode=='down':
      self.resample = nn.AvgPool1d(kernel_size=2, stride=2)
    elif mode=='up':
      self.resample = Upsample1D()
    else:
      self.resample = None
    
  def forward(self, x, dx):
    x = self.r( self.a(x,dx) )
    x = self.r( self.b(x,dx) )
    if self.resample: 
      x = self.resample(x)
    return x


class unet_1d(nn.Module):
  def __init__(self, init_features=8):
    super(unet_1d, self).__init__()
    self.e01 = block(1, init_features, mode='down', ps=[-1,0])
    self.e12 = block(init_features*1, init_features*2, mode='down', ps=[0])
    self.e23 = block(init_features*2, init_features*4, mode='down', ps=[0])
    self.bottleneck = block(init_features*4, init_features*8, ps=[0])
    self.d32 = block(init_features*8, init_features*16, mode='up', ps=[0])
    self.d21 = block(init_features*34, init_features*16, mode='up', ps=[0])
    self.d10 = block(init_features*33, init_features*16, mode='up', ps=[0])
    self.pred = scale_aware_conv1d_alt(init_features*32, 1, ps=[0,1])
  
  def forward(self, x, dx):
    f1 = self.e01(x, dx)    # init_features
    f2 = self.e12(f1, dx)   # init_features * 2
    f3 = self.e23(f2, dx)   # init_features * 4
    
    f3 = self.bottleneck(f3, dx)                 # features at stride 8
    f2 = torch.cat( [self.d32(f3, dx), f2], dim=1 )
    f1 = torch.cat( [self.d21(f2, dx), f1], dim=1 )
    f0 = self.d10(f1, dx)
    return self.pred(f0, dx)
    
#####################################
# Causal space and time aware model #
#####################################

class cstam(nn.Module):
  def __init__(self):
    super(cstam, self).__init__()
    self.space_model1 = unet_1d()
    self.space_model2 = unet_1d()
    
  def forward(self, init_state, dx, dt, N):
    # Process state
    state = init_state
    states = []
    for t in range(N):
      state = state + torch.clip(self.space_model1(state, dx),-3,3) * torch.clip(self.space_model2(state, dx),-3,3) * dt # have to bound due to dx term
      states.append(state)
    return torch.stack(states, dim=3)
    
      
      

    
