import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

# This repo
sys.path.append(str(Path(__file__).resolve().parent / 'network' ))
sys.path.append(str(Path(__file__).resolve().parent / 'data'))
import burgers_dataset
from unet_1d import cstam

##################################################################
# Options for running the script- should probably turn into args #
##################################################################
retrain = True
reevaluate = True
use_cuda = torch.cuda.is_available()
initial_weights = None # 'weights/unet_1.pth'

###############################################
# Create dataloader around our dataset object #
###############################################
train_dataset = burgers_dataset.Dataset()
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0) 

###############################################
# Create Unet and optimizer to solve params   #
###############################################
model = cstam()
if use_cuda: model = model.cuda()
if initial_weights is not None:
  model.load_state_dict(torch.load(initial_weights))  # load a state dictionary for the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-9)
loss_fcn = torch.nn.MSELoss()
iter_size = 3

###############################################
# Optimize the Unet model parameters          #
###############################################
if retrain:
  losses = []
  residuals = []
  # Loop over the dataset
  for epoch in range(5):
    for i, data in enumerate(train_loader, 0):
      u0, U = data             # b, p are batch x t x c x l
      if use_cuda: u0, U = u0.cuda(), U.cuda()
      if i % iter_size == 0:
        optimizer.zero_grad()
      preds = model(u0, train_dataset.dx, train_dataset.dt, train_dataset.N_t)
      loss = loss_fcn(preds, U)
      loss.backward()
      print( 'Epoch: {}, Iteration: {}, loss: {}'.format(epoch, i, loss.item() ) )
      if i % iter_size == 0:
        optimizer.step()
      losses.append(loss.item())

      # Change domain discretization before next iteration
      new_Nx = 128 * np.random.randint(3,6)
      new_Nt = 40 #np.random.randint(30, 50)
      train_dataset.domain_change(new_Nx,new_Nt)

    PATH = 'weights/unet_{}.pth'.format(epoch + 1)
    torch.save(model.state_dict(), PATH)

###############################################
# Make some qualitative plots to see what     #
###############################################
train_dataset.domain_change(128*4,40)
with torch.no_grad():
  for i in range(10):
    u0, U = train_dataset[0]
    if use_cuda: u0, U = u0.cuda(), U.cuda()
    preds = model(u0[None], train_dataset.dx, train_dataset.dt, train_dataset.N_t)[0]
    print(preds.shape)

    # Look at end of shape...
    Ufinal = U[0,:,-1].cpu().numpy()
    Uinit = U[0,:,0].cpu().numpy()
    pfinal = preds[0,:,-1].cpu().numpy()
    pinit = preds[0,:,0].cpu().numpy()
    plt.plot(Ufinal,'b-')
    plt.plot(pfinal,'r-')
    plt.plot(Uinit,'b*')
    plt.plot(pinit,'rx')
    plt.show()
  
###############################################
# Plot errors for different dx, dt values     #
###############################################
# Try 8-13 (256-8192 dx steps)
# Try 20-100 for different T steps?
# y-axis=mean-error, x-axis=N_x 

test_dataset = burgers_dataset.Dataset(cnt=100,test=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0) 

Nxs = np.array([128*xx for xx in range(2,9)])
Nts = np.array( list(range(20,70,10)) )
all_errs = np.zeros((Nxs.shape[0], Nts.shape[0]))

if reevaluate:
  for ii, new_Nx in enumerate(Nxs):
    for jj, new_Nt in enumerate(Nts):
      # Configure test dataset
      test_dataset.domain_change(new_Nx, new_Nt)

      # Go through test dataset
      with torch.no_grad():
        errs = []
        for i, data in enumerate(test_loader, 0):
          u0, U = data             # b, p are batch x t x c x l
          preds = model(u0, test_dataset.dx, test_dataset.dt, test_dataset.N_t)
          loss = loss_fcn(preds, U)
          errs.append(loss.item())
        
        # Average errors
        all_errs[ii,jj] = np.mean(errs)
  np.save('artifacts/all_errs.npy', all_errs)

all_errs = np.load('artifacts/all_errs.npy')
all_Nts, all_Nxs = np.meshgrid(Nts, Nxs)
data = np.stack([all_Nxs, all_Nts, all_errs],axis=2).reshape(-1,3)
import pandas as pd
df = pd.DataFrame(data,columns=['Nx','Nt','err'])
df['in_domain'] = (df['Nx']>=384)*(df['Nx']<=512)*(df['Nt']>=30)*(df['Nt']<=50)
df['dx'] = 10.0 / df['Nx']
df['dt'] = 1.0 / df['Nt']

# Plot better
clrs = list(mcolors.BASE_COLORS.keys())
for ii in range(Nxs.shape[0]):
  # Plot all dts as striped
  sub_df = df[df['Nx']==Nxs[ii]]
  dx = np.array(sub_df['dx'])[0].item()
  format = '--{}'.format(clrs[ii])
  plt.plot(sub_df['dt'], sub_df['err'],format, label='dx={:.3f}'.format(dx))

  # Plot over with solid anything that is in the training domain
  sub_sub_df = sub_df[sub_df['in_domain']]
  format = '-{}'.format(clrs[ii])
  plt.plot(sub_sub_df['dt'],sub_sub_df['err'], format)

plt.suptitle('Mean Square Error vs. Discretization')
plt.title('Solid lines indicate discretizations seen at training time')
plt.xlabel('dt')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plots...
for ii in range(Nxs.shape[0]):
  plt.plot(Nts, all_errs[ii,:], label='N_x={}'.format(Nxs[ii]))
plt.legend()
plt.ylabel('Mean squared error')
plt.xlabel('Number of time steps')
plt.show()

