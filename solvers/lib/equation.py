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
            if not BC_x[0] in ["same", "per", "periodic"]:
                cls.E[0,0]=1
                cls.e[0]=BC_x[0]
            elif BC_x[0].lower() in ["per", "periodic"]:
                cls.E[0,0]=1

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][0,:] = np.zeros_like( cls.gradient_matrices[i][0,:] )

        # BC RHS - x
        if not BC_x[-1]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_x[-1] in ["same", "per", "periodic"]:
                    cls.E[-1,-1]=1
                    cls.e[-1]=BC_x[-1]
            elif BC_x[-1].lower() in ["per", "periodic"]:
                cls.E[0,-1]=-1

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][-1,:] = np.zeros_like( cls.gradient_matrices[i][-1,:] )
        
        # BC LHS - dx
        if not BC_dx[0]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_dx[0] in ["same", "per", "periodic"]:
                cls.E[0,:]=cls.gradient_matrices[0][0,:]
                cls.e[0]=BC_dx[0]
            elif BC_x[0].lower() in ["per", "periodic"]:
                cls.E[0,0]=1

            for i in range( len( cls.gradient_matrices ) ):
                cls.gradient_matrices[i][0,:] = np.zeros_like( cls.gradient_matrices[i][0,:] )
            
        # BC RHS - dx
        if not BC_dx[-1]==None:
            bc_count+=1
            # ie: Hold the same values at this location
            if not BC_dx[-1] in ["same", "per", "periodic"]:
                cls.E[-1,:]=cls.gradient_matrices[0][-1,:]
                cls.e[-1]=BC_dx[-1]
            elif BC_x[0].lower() in ["per", "periodic"]:
                cls.E[0,-1]=-1

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
        BC_dx2 = BCs[2]

        # Call the parent class call method
        if cls.viscid:
            super().__call__(x, u, nu, BC_x=BC_x, BC_dx=BC_dx, BC_dx2=BC_dx2 )
        else:
            super().__call__(x, u, nu, BC_x=BC_x, BC_dx=BC_dx )

        # Set up A-matrix - ie: 2nd derivative
        if cls.viscid and not nu==0:
            cls.A = nu * cls.gradient_matrices[1]

        # Set up B-matrix
        cls.B = cls.gradient_matrices[0].todia()
        cls.f = u*u/2

        # Sum to time derivative
        du_dt = -cls.B.dot(cls.f) + cls.E.dot(cls.e)
        if cls.viscid and not nu==0:
            du_dt += -cls.A.dot(u)

        return du_dt
    
class stewartLandau_eqn(eqn_problem):
    """
        This is the object that allows us to find the solution of the Stewart-Landau equation. This
    was specifically developed for MAT-5323 at the University of Texas at San Antonio, Assignment
    3.

        This takes the equations:

    $$
    \frac{dr}{dt} = \mu r(t) - r^3 (t) 
    $$
    $$
    \frac{d\theta}{dt} = \gamma - \beta r^2 (t) 
    $$

        In this case, $t$ is the parameter or time. 

    """

    def __init__(self, stepping="explicit", SNR=0.0 ):
        """
            Initialize the Stewart-Landau Equation problem.
        
        """

        # Initialize from eqn_problem
        super().__init__(2, 2, stepping=stepping, max_derivative=1)

        # Store SNR
        self.SNR = SNR

    def __call__(cls, f, coeffs ):
        """
            Set of Differential equations to solve the Stewart-Landau Equation.

            The equation is set up in the following format:

        **Explicit methods**

        <du/dt>=[dt/dt, dtheta/dt]=[mu*r - r^3, gamma - beta*r^2]

        Where

        Args:
            f (float, numpy ndarray):   The function values of the Stewart-Landau equation. Will
                                        be a 2D array of the form [r, theta].

            coeffs (float): A list, tuple, or array of the coefficients in the Stewart-Landau 
                                        equation. The order is [mu, gamma, beta].

        """

        # Pull the coefficients
        mu = coeffs[0]
        gamma = coeffs[1]
        beta = coeffs[2]

        # Pull the boundary conditions
        r = f[0]
        theta = f[1]

        # Create the time step
        dr_dt = mu*r - r**3
        dtheta_dt = gamma - beta*r**2
        du_dt = np.array( [dr_dt, dtheta_dt] ) * ( np.ones(2) + cls.SNR*np.random.randn(2) )

        return du_dt

#==================================================================================================
#
#   Discrete Wavelet Transform Equations
#
#==================================================================================================

def wavelet_dilation( lower_wavelet_in, upper_wavelet_in, dilation_level ):
    """
        This function takes the lower and upper wavelet coefficients and dilates them by a given 
    number of levels. This is used to create the wavelet transform.

    Args:
        lower_wavelet_in (float, numpy 1Darray):    The wavelet in the lower level, or closer to 
                                                    the scaling function.

        upper_wavelet_in (float, numpy 1Darray):    The wavelet in the upper level, or closer to 
                                                    the highest level of wavelet function.

        dilation_level (int):    The number of levels to dilate the wavelet by.

    """

    # Import PyWavelets, we do this here so other 1D equations don't need to import it
    import pywt

    # Calculate the factors for the dilation
    dilation_factor = 2**dilation_level
    data_length = len( lower_wavelet_in ) * dilation_factor
    #print(f"Data length is {data_length}")

    # Define the lower level wavelet
    upper_wavelet = np.zeros( data_length )
    upper_wavelet[::dilation_factor] = upper_wavelet_in
    upper_wavelet *= 2**(dilation_level/2)

    # Define the upper level wavelet
    lower_wavelet = np.zeros( data_length )
    index_offset = (data_length-len(lower_wavelet_in))//2
    #print(f"Index offset:\t{index_offset}")
    lower_wavelet[index_offset:-index_offset] = lower_wavelet_in
    #lower_wavelet *= 2**(dilation_level/2)

    return lower_wavelet, upper_wavelet

def wavelet_contraction( lower_wavelet_in, upper_wavelet_in, contraction_level ):
    """
        This function takes the lower and upper wavelet coefficients and contracts them by a given
    number of levels. This is used to create the wavelet transform.

    Args:
        lower_wavelet_in (float, numpy 1Darray):    The wavelet in the lower level, or closer to 
                                                    the highest level of wavelet function.

        upper_wavelet_in (float, numpy 1Darray):    The wavelet in the upper level, or closer to 
                                                    the scaling function.

        contract_level (int):    The number of levels to contract the wavelet by.

    """

    # Import PyWavelets, we do this here so other 1D equations don't need to import it
    import pywt

    # Calculate the factors for the dilation
    contraction_factor = 2**contraction_level
    data_length = len( lower_wavelet_in )
    #print(f"Data length is {data_length}")

    # Define the lower level wavelet
    lower_wavelet = np.zeros( data_length )
    #print(f"Index offset:\t{(data_length-len(lower_wavelet_in)//contraction_factor)//2}")
    index_offset = (data_length-len(lower_wavelet_in)//contraction_factor)//2
    #print(f"Wavelet input:\t{lower_wavelet_in[::contraction_factor]}")
    lower_wavelet[index_offset:-index_offset] = lower_wavelet_in[::contraction_factor]
    #print(f"Lower wavelet shape: {lower_wavelet.shape}")
    #print(f"Lower wavelet: {lower_wavelet}")
    lower_wavelet *= 2**(-contraction_level/2)

    # Define the upper level wavelet
    upper_wavelet = np.zeros( data_length )
    upper_wavelet = np.array( upper_wavelet_in )
    #upper_wavelet *= 2**(-contraction_level/2)

    return lower_wavelet, upper_wavelet



class wavelet_eqn(eqn_problem):
    """
        This object is the parent object for DWT-based equations.

    """

    def __init__(self, spatial_order=2, spatialBC_order=None, stepping="explicit", max_derivative=4 ):
        """
            Initialize the DWT equation problem.

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
        # Import PyWavelets, we do this here so other 1D equations don't need to import it
        import pywt

        # Set up boundary condition order
        if spatialBC_order is None:
            spatialBC_order = spatial_order

        # Initialize from eqn_problem
        super().__init__(spatial_order, spatialBC_order, stepping=stepping, max_derivative=max_derivative)

    def kernel( cls, x, wt_family, levels=None, wt_mode="symmetric" ):
        """
            This method creates the kernels that linearize the wavelet transform.

        Args:
            x (float, numpy ndarray):   This is the 1D spatial domain that the wavelet transform 
                                        will be applied to.

            wt_family (string): The wavelet family that will be used in the DWT.

            levels (int, optional): The number of levels that the wavelet transform will be applied
                                        to. Defaults to None, which will use the maximum number of
                                        levels that can be applied to the data.

        """
        # Import PyWavelets, we do this here so other 1D equations don't need to import it
        import pywt
        import scipy.sparse as spsr

        # Set the wavelet family
        cls.wt_family = wt_family
        cls.wt = pywt.Wavelet( wt_family )

        # Set the domain
        cls.x = x
        cls.dx = np.mean( np.gradient( cls.x ) )

        # Set the levels
        cls.max_levels = pywt.dwt_max_level( len(x), cls.wt.dec_len )
        if levels is None:
            cls.levels = cls.max_levels
        else:
            cls.levels = levels

        # Set the size of the DWT
        coeffs_samples = pywt.wavedec( np.zeros_like(x), cls.wt_family, mode=wt_mode, level=cls.levels )
        cls.level_sizes = []
        for i in range( len( coeffs_samples ) ):
            cls.level_sizes += [ coeffs_samples[i].shape[0] ]
        cls.level_sizes = np.array( cls.level_sizes, dtype=int )

        # Set the start/end indices of the wavelet transform
        cls.end_indices = np.cumsum( cls.level_sizes )
        cls.start_indices = np.roll( cls.end_indices, 1 )
        cls.start_indices[0] = 0

        


        


    def autokernel(cls ):


        #
        # Set the autokernel or Gram matrix
        #
        cls.M = spsr.csr_matrix( ( np.sum(cls.level_sizes), np.sum(cls.level_sizes) ), dtype=float )
        # Iterate over the row sets that pertain to the levels
        for li in range( cls.levels+1 ):
            #print(f"Row level index {li} from rows [{cls.start_indices[li]}, {cls.end_indices[li]}]")

            # Select the row-based wavelet shape
            if cls.start_indices[li]==0:
                #print(f"\tSelecting scaling function for row")
                row_level = 1
                row_wavelet = cls.wt.wavefun( level = row_level )[0][::-1]
            else:
                #print(f"\tSelecting wavelet function for row")
                row_level = li
                row_wavelet = cls.wt.wavefun( level = row_level )[1][::-1]
            #print(f"\t\tRow wavelet:\t{row_wavelet}")

            for lj in range( cls.levels+1 ):
                #print(f"\tColumns level index {lj} from rows [{cls.start_indices[lj]}, {cls.end_indices[lj]}]")

                # Select the column-based wavelet shape
                if cls.start_indices[lj]==0:
                    #print(f"\t\tSelecting scaling function for column")
                    col_level = 1
                    col_wavelet = cls.wt.wavefun( level = col_level )[0]
                else:
                    #print(f"\t\tSelecting wavelet function for column")
                    col_level = lj
                    col_wavelet = cls.wt.wavefun( level = col_level )[1]
                #print(f"\t\tColumn wavelet:\t{col_wavelet}")

                """
                # Calculate the level difference
                level_diff = row_level - col_level
                print(f"\t\tLevel difference: {level_diff}")
                row_wavelet_hold = row_wavelet.copy()
                col_wavelet_hold = col_wavelet.copy()
                if np.abs(level_diff)>0:
                    col_wavelet = np.array( col_wavelet )
                    #col_wavelet *= 2**(-level_diff/2)
                    if level_diff>0:
                        print(f"\t\t\tDilation needed")
                        _, col_wavelet = wavelet_dilation( row_wavelet, col_wavelet, np.abs( level_diff ) )
                        col_wavelet *= 2**(-level_diff/2)
                    else:
                        print(f"\t\t\tContraction needed")
                        _, col_wavelet = wavelet_contraction( row_wavelet, col_wavelet, np.abs( level_diff ) )
                        col_wavelet *= 2**(level_diff/2)

                    print(f"\t\t\tNew row wavelet shape: {row_wavelet}")
                    print(f"\t\t\tNew column wavelet shape: {col_wavelet}")
                #"""


                # Calculate raw convolution
                convolve_factor = (2 ** -( ( row_level + col_level ) / 2 ))
                #print(f"\t\tRow wavelet:\t{row_wavelet}")
                #print(f"\t\tColumn wavelet:\t{col_wavelet}")
                #print(f"\t\tConvolve factor: {convolve_factor}")
                raw_c = np.convolve( row_wavelet, col_wavelet, mode="valid" ) 
                #print(f"\t\tRaw convolution shape: {raw_c.shape}")
                #print(f"\t\tRaw convolution: {raw_c}")
                #print(f"\t\tRaw convolution sum: {np.sum(raw_c)}")

                # Calculate the inner product
                #print(f"\t\tSplit is {np.ceil( ( row_level + col_level )/2).astype(int)}")
                inner = raw_c[::np.ceil( ( row_level + col_level )/2).astype(int)] * convolve_factor
                #inner = inner[cut:-cut]
                #print(f"\t\tInner product cut: {cut}")
                #print(f"\t\tInner product shape: {inner.shape}")
                #print(f"\t\tInner product: {inner}")

                for ii in range( cls.start_indices[li], cls.end_indices[li] ):
                    #print(f"\t\tRow {ii}:")

                    # Place the inner product in the row
                    #print(f"\t\t\tHas length {cls.level_sizes[lj]} from lj {lj}")
                    row = np.zeros( cls.level_sizes[lj] )
                    row[:len(inner)]=inner
                    row = np.roll( row, ii-cls.start_indices[li] )
                    #print(f"\t\t\tRow:\t{row}")
                    
                    # Place the inner product in the matrix
                    #print(f"\t\t\tPlacing inner product in matrix at {ii}, {cls.start_indices[lj]}:{cls.end_indices[lj]}")
                    cls.M[ii, cls.start_indices[lj]:cls.end_indices[lj]] = row

                #row_wavelet = row_wavelet_hold
                #col_wavelet = col_wavelet_hold


    def gradientKernel(cls ):


        #
        # Set the autokernel or Gram matrix
        #
        cls.D = spsr.csr_matrix( ( np.sum(cls.level_sizes), np.sum(cls.level_sizes) ), dtype=float )
        # Iterate over the row sets that pertain to the levels
        for li in range( cls.levels+1 ):
            print(f"Row level index {li} from rows [{cls.start_indices[li]}, {cls.end_indices[li]}]")

            # Select the row-based wavelet shape
            if cls.start_indices[li]==0:
                print(f"\tSelecting scaling function for row")
                row_level = 1
                row_wavelet = cls.wt.wavefun( level = row_level )[0][::-1]
            else:
                print(f"\tSelecting wavelet function for row")
                row_level = li
                row_wavelet = cls.wt.wavefun( level = row_level )[1][::-1]
            print(f"\t\tRow wavelet:\t{row_wavelet}")

            for lj in range( cls.levels+1 ):
                print(f"\tColumns level index {lj} from rows [{cls.start_indices[lj]}, {cls.end_indices[lj]}]")

                # Select the column-based wavelet shape
                if cls.start_indices[lj]==0:
                    print(f"\t\tSelecting scaling function for column")
                    col_level = 1
                    col_wavelet = cls.wt.wavefun( level = col_level )[0]
                else:
                    print(f"\t\tSelecting wavelet function for column")
                    col_level = lj
                    col_wavelet = cls.wt.wavefun( level = col_level )[1]
                gradient_factor = (2**(col_level-1))
                print(f"\t\tGradient factor: {gradient_factor}")
                col_wavelet = np.gradient( col_wavelet, cls.dx / gradient_factor )
                print(f"\t\tColumn wavelet:\t{col_wavelet}")

                # Calculate raw convolution
                convolve_factor = (2 ** -( ( row_level + col_level ) / 2 ))
                print(f"\t\tRow wavelet:\t{row_wavelet}")
                print(f"\t\tColumn wavelet:\t{col_wavelet}")
                print(f"\t\tConvolve factor: {convolve_factor}")
                raw_c = np.convolve( row_wavelet, col_wavelet, mode="valid" ) 
                print(f"\t\tRaw convolution shape: {raw_c.shape}")
                print(f"\t\tRaw convolution: {raw_c}")
                print(f"\t\tRaw convolution sum: {np.sum(raw_c)}")

                # Calculate the inner product
                print(f"\t\tSplit is {np.ceil( ( row_level + col_level )/2).astype(int)}")
                inner = raw_c[::np.ceil( ( row_level + col_level )/2).astype(int)] * convolve_factor
                #inner = inner[cut:-cut]
                #print(f"\t\tInner product cut: {cut}")
                print(f"\t\tInner product shape: {inner.shape}")
                print(f"\t\tInner product: {inner}")

                for ii in range( cls.start_indices[li], cls.end_indices[li] ):
                    print(f"\t\tRow {ii}:")

                    # Place the inner product in the row
                    print(f"\t\t\tHas length {cls.level_sizes[lj]} from lj {lj}")
                    row = np.zeros( cls.level_sizes[lj] )
                    row[:len(inner)]=inner
                    row = np.roll( row, ii-cls.start_indices[li] )
                    print(f"\t\t\tRow:\t{row}")
                    
                    # Place the inner product in the matrix
                    print(f"\t\t\tPlacing inner product in matrix at {ii}, {cls.start_indices[lj]}:{cls.end_indices[lj]}")
                    cls.D[ii, cls.start_indices[lj]:cls.end_indices[lj]] = row

                #row_wavelet = row_wavelet_hold
                #col_wavelet = col_wavelet_hold  




