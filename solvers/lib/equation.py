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

        <du/dt>=[A]<u>-[B]<f>+[E]<e>

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

def samplesToCoeffsDWT( N_samples, N_levels, support, verbosity=0 ):
    """
        This determines which samples of the original signal are represented by the coefficients
    for each level of the DWT.

        Note that this only pertains to a 1D DWT of multiple levels.

    Args:
        N_samples (int):    The number of samples in the original signal.

        N_levels (int): The number of levels in the DWT.

        support (int): The number of samples in the wavelet used in the DWT.

    Returns:
        coeff_list (list, int): A list of integers where each integer represents the number of coefficients at each level of the DWT. In the format:

                                [ level ][ coefficient index, sample index ]

    """

    coeff_list = []
    for i in np.arange( N_levels ):
        # The number of coefficients at this level
        N_samples_perLevel = np.ceil( N_samples / ( 2 ** i ) ).astype(int)

        # The number of samples represented by each coefficient
        N_samples_per_coeff = int( support * ( 2 ** i ) )

        # The number of coefficients that can be represented at this level
        if i==0:
            data_length=N_samples
        else:
            data_length=N_coeffs_per_level
        N_coeffs_per_level = np.floor( ( data_length + support - 1 )/2 )
        N_coeffs_per_level = int( N_coeffs_per_level )

        if verbosity > 0:
            print( f"Level {i}: {N_samples_perLevel} samples, each coefficient representing {N_samples_per_coeff} samples, totaling {N_coeffs_per_level} coefficients" )

        #
        #   Produce the list of indices from the original signal that are represented by each coefficient
        # 
        coeff_list_atLevel = np.zeros( ( N_coeffs_per_level, N_samples_per_coeff ), dtype=int )
        if verbosity > 1:
            print( f"\tcoeff_list_atLevel.shape: {coeff_list_atLevel.shape}" )
        for j in np.arange( coeff_list_atLevel.shape[0] ):
            coeff_list_atLevel[j, :] = np.arange( j * (2**(i+1)), j * (2**(i+1)) + N_samples_per_coeff )-(2**(i))-1

        # Check if the last coefficient goes over for over samples and shift as needed
        #"""
        if np.max( coeff_list_atLevel ) >= N_samples:
            difference = np.max( coeff_list_atLevel ) - N_samples
            if verbosity > 0:
                print( f"\tLast coefficient goes over the number of samples, shifting by {difference//2}" )
            coeff_list_atLevel = coeff_list_atLevel - difference // 2
        #"""

        coeff_list += [ coeff_list_atLevel ]


    return coeff_list

def lineDomainDWT( domain, N_levels, support, verbosity=0, Truncate_domain=True ):
    """
        This function converts a 1D domain into the equivalent domain represented by the DWT coefficients.
    

    Args:
        domain (float, array):  The original domain to be converted.

        N_levels (int): The number of levels in the DWT.

        support (int): The number of samples in the wavelet used in the DWT.

    Returns:
        DWT_domain (float, list): The domain represented by the DWT coefficients. Will be in format:
                                    [ level ][ coefficient index ]

                                    
    """

    # Pull the coefficients for the domain
    coeffs = samplesToCoeffsDWT( domain.shape[0], N_levels, support )

    # Initialize the DWT domain
    DWT_domain = []
    for l in np.arange( N_levels ):
        if verbosity > 0:
            print(f"Level {l}:")
        DWT_domain_atLevel = np.zeros( coeffs[l].shape[0] )
        for c in np.arange( coeffs[l].shape[0] ):
            if verbosity > 0:
                print(f"\tCoefficient {c}:\t{coeffs[l][c]}")

            # Correct for lower bound
            filtered_coeffs = coeffs[l][c][coeffs[l][c]>=0]

            # Correct for upper bound
            filtered_coeffs = filtered_coeffs[filtered_coeffs<domain.shape[0]]

            # Add the domain represented by this coefficient
            domain_at_coeff = domain[ filtered_coeffs ]
            if verbosity > 1:
                print(f"\t\tDomain at coefficient {c}:\t{domain_at_coeff}")

            if not Truncate_domain:
                # Calculate the number of coefficients kept and dropped
                number_kept = len(filtered_coeffs)
                number_dropped = len(coeffs[l][c])-number_kept


                if not number_dropped==0:
                    # Figure out if this is the LHS or RHS
                    LHS_truncate = any( x<0 for x in coeffs[l][c] )
                    RHS_truncate = any( x>=domain.shape[0] for x in coeffs[l][c] )

                    og_right_location = domain_at_coeff[-1]
                    og_left_location = domain_at_coeff[0]
                    centroid_location = np.mean( domain_at_coeff )
                    dx = np.mean( np.gradient( domain_at_coeff ) )

                    if LHS_truncate:
                        left_location = og_left_location - dx * number_dropped
                        domain_at_coeff = np.arange( left_location, og_right_location+dx, dx )

                    elif RHS_truncate:
                        right_location = og_right_location - dx * number_dropped
                        domain_at_coeff = np.arange( og_left_location, right_location+dx, dx )
                
            # Calculate the centroid for the domain at the coefficient
            DWT_domain_atLevel[c] = np.mean( domain_at_coeff )

        DWT_domain += [ DWT_domain_atLevel ]

    return DWT_domain

class wavelet_eqn(eqn_problem):
    """
        This object is the parent object for DWT-based equations.

    """

    def __init__(self, spatial_order=2, spatialBC_order=None, stepping="explicit", max_derivative=2, N_levels=1, wavelet="db2", signal_extension="zero", gradient_construction="split" ):
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
                                                object. Default value is 2.

            N_levels (int): The number of levels in the DWT.

            wavelet (str):  The type of wavelet to be used in the DWT. Defaults to "db2", or Daubechies
                                with 2 vanishing moments.

            signal_extension (str):    The type of signal extension to be used in the DWT. Defaults
                                        to "zero". Other options include "symmetric", "periodic", 
                                        etc. Check the PyWavelets documentation for more details.

            gradient_construction (str):  The method to construct the gradient in wavelet space. 
                                            The valid options are:

                                        -"*split":  This method splits the gradient across the 
                                                    detial and approximation spaces. This assumes
                                                    that \frac{\partial d}{\partial x}=0 and
                                                    \frac{\partial \tilde{\phi}{\partial x}*\phi=0.

                                        -"modified": This method modifies the gradient to across 
                                                    the detail space. This assumes that
                                                    \frac{\partial d}{\partial x}=0.

                                        -"full": This method calculates the full gradient across
                                                    both the detail and approximation spaces.

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

        # Store wavelet data
        self.N_levels = N_levels
        self.wavelet = wavelet
        self.support = pywt.Wavelet( wavelet ).dec_len
        self.signal_extension = signal_extension

        # Store gradient construction method
        self.gradient_construction = gradient_construction
        self.max_derivative = max_derivative   

    def waveshape_precompute(cls, diff_method="cd", enforce_real=True ):
        """
            This method precomputes the wavelet shapes, their derivatives, and their inner 
        products. This includes the DFT representation of the wavelet functions.

        Args:
            diff_method (str, optional):    The method to calculation the derivative of the 
                                            wavelet. The valid options are:

                                            "central difference", "central", "cd":
                                                Use NumPy's gradient method.

                                            "spectral", "fourier", or "frequency":
                                                User spectral differencing.

        """
        import pywt

        #=============================================================
        #
        #   Precompute the wavelet shapes
        #
        #=============================================================

        # Initialize storage values
        cls.wavelet_shapes = {}

        # Calculate the approximation function shape
        cls.wavelet_shapes["phi_decomp"] = pywt.Wavelet( cls.wavelet ).dec_lo
        cls.wavelet_shapes["phi_rebuild"] = pywt.Wavelet( cls.wavelet ).rec_lo

        # Calculate the wavelet function shape
        cls.wavelet_shapes["psi_decomp"] = pywt.Wavelet( cls.wavelet ).dec_hi
        cls.wavelet_shapes["psi_rebuild"] = pywt.Wavelet( cls.wavelet ).rec_hi

        cls.support = len( cls.wavelet_shapes["phi_decomp"] )

        # Store the original keys
        og_keys = list( cls.wavelet_shapes.keys() )
        cls.og_keys = og_keys

        # Calculate the padded shapes
        n = len(cls.wavelet_shapes["phi_decomp"]) + len(cls.wavelet_shapes["phi_rebuild"]) - 1
        n_padded = 2**int(np.ceil(np.log2(n)))+2*(cls.max_derivative-1)
        cls.n_padded = n_padded
        n_diffference = n_padded - len(cls.wavelet_shapes["phi_decomp"])
        for k in og_keys:
            cls.wavelet_shapes[k] = np.array( cls.wavelet_shapes[k] )
            cls.wavelet_shapes[f"{k}_padded"] = np.zeros( n_padded )
            cls.wavelet_shapes[f"{k}_padded"][n_diffference//2:-n_diffference//2] = cls.wavelet_shapes[k]

        #=============================================================
        #
        #   Precompute the wavelet shape DFT
        #
        #=============================================================

        # Initialize storage values
        cls.wavelet_shapes_DFT = {}

        # Calculate the DFT of the wavelet shapes from the padded shapes
        for k in og_keys:
            cls.wavelet_shapes_DFT[k] = np.roll( np.fft.fft( cls.wavelet_shapes[f"{k}_padded"] ), n_padded//2 )

        # Initialize storage of corresponding wavenumbers
        cls.wavelet_shapes_wavenumbers_normalized = np.roll( np.fft.fftfreq( n_padded ), n_padded//2 )

        #=============================================================
        #
        #   Precompute the wavelet shape derivatives
        #
        #=============================================================

        # Initialize storage values
        cls.wavelet_shapes_deriv = {}
        cls.wavelet_deriv_DFT = {}

        # Calculate the derivatives of the wavelet shapes
        for k in og_keys:
            if diff_method.lower() in ["central difference", "central", "cd"]:
                cls.wavelet_shapes_deriv[k] = np.gradient( cls.wavelet_shapes[f"{k}_padded"], edge_order=1 )
                # TODO: Add in higher order derivatives as needed
                cls.wavelet_shapes_deriv[f"{k}_2ndDeriv"] = np.gradient( cls.wavelet_shapes_deriv[k], edge_order=1 )
            elif diff_method.lower() in ["spectral", "fourier", "frequency"]:
                # First derivative
                cls.wavelet_deriv_DFT[k] = 1j * cls.wavelet_shapes_wavenumbers_normalized * cls.wavelet_shapes_DFT[k]
                cls.wavelet_shapes_deriv[k] = np.fft.ifft( cls.wavelet_deriv_DFT[k] )
                
                # Second derivative
                cls.wavelet_deriv_DFT[f"{k}_2ndDeriv"] = 1j * cls.wavelet_shapes_wavenumbers_normalized * cls.wavelet_deriv_DFT[k]
                cls.wavelet_shapes_deriv[f"{k}_2ndDeriv"] = np.fft.ifft( cls.wavelet_deriv_DFT[f"{k}_2ndDeriv"] )

        if diff_method.lower() in ["spectral", "fourier", "frequency"]:
            if enforce_real:
                for k in list( cls.wavelet_shapes_deriv.keys() ):
                    cls.wavelet_shapes_deriv[k] = cls.wavelet_shapes_deriv[k].real

        #=============================================================
        #
        #   Precompute the wavelet shape derivatives' convolution
        #
        #=============================================================

        # Initialize storage values
        cls.wavelet_shapes_deriv_convolution = {}
        cls.wavelet_shapes_deriv_convolution_raw = {}

        # Calculate the convolution of the derivatives of the wavelet shapes
        for k in ["phi", "psi"]:
            #raw_convolution = np.convolve( cls.wavelet_shapes_deriv[f"{k}_decomp"], cls.wavelet_shapes[f"{k}_rebuild"], mode="valid" )
            raw_convolution = np.convolve( cls.wavelet_shapes_deriv[f"{k}_rebuild"], cls.wavelet_shapes[f"{k}_decomp"], mode="valid" )
            cls.wavelet_shapes_deriv_convolution_raw[f"{k}'*{k}"] = raw_convolution
            cls.wavelet_shapes_deriv_convolution[f"{k}'*{k}"] = raw_convolution[::2]

        # Calculate the convolution of the 2nd derivatives of the wavelet shapes
        for k in ["phi", "psi"]:
            #raw_convolution = np.convolve( cls.wavelet_shapes_deriv[f"{k}_decomp_2ndDeriv"], cls.wavelet_shapes[f"{k}_rebuild"], mode="valid" )
            raw_convolution = np.convolve( cls.wavelet_shapes_deriv[f"{k}_rebuild_2ndDeriv"], cls.wavelet_shapes[f"{k}_decomp"], mode="valid" )
            cls.wavelet_shapes_deriv_convolution_raw[f"{k}''*{k}"] = raw_convolution
            cls.wavelet_shapes_deriv_convolution[f"{k}''*{k}"] = raw_convolution[::2]

    def domain_initialization(cls, x_domain ):
        """
            This method takes the domain the wavelet equation is being solved on and initializes
        the domain that the wavelet coefficients are present on.

        Args:
            x_domain (numpy 1Darray - float): The spatial domain that the wavelet equation is being
                                                solved on.

        """
        # Store the original domain
        cls.x_domain = x_domain
        
        # Calculate and store the DWT domain
        cls.DWT_domain = lineDomainDWT( x_domain, cls.N_levels, cls.support )[::-1]

    def wavelet_initialization(cls, u_0, storage_level=0 ):
        """
            This method takes the initial condition and converts it to the wavelet coefficients.
        This allows the object to initialize the solution in the wavelet space.

        Args:
            u_0 (numpy 1Darray - float):    The initial condition to be converted to wavelet space.

            storage_level (int, optional):  The amount of things to store in the object.

        """
        # Import PyWavelets, we do this here so other 1D equations don't need to import it
        import pywt

        # Check that the initial condition is the same size as the domain
        if not u_0.shape[0] == cls.x_domain.shape[0]:
            raise ValueError("Initial condition must be the same size as the domain.")
        
        # Initialize the function and store
        cls.u = []
        cls.u += [ u_0 ]
        """
            Note that we are storing the function in the format:

        u[time index][x-domain index]

            Where the x-domain index listing represents a NumPy array.

        """

        # Perform the DWT to get the coefficients
        if cls.N_levels==1:
            coeffs = pywt.dwt( cls.u[0], cls.wavelet, mode=cls.signal_extension )
        elif cls.N_levels>1:
            coeffs = pywt.wavedec( cls.u[0], cls.wavelet, mode=cls.signal_extension, level=cls.N_levels )
        else:
            raise ValueError("N_levels must be at least 1.")
        if storage_level>0:
            cls.raw_coeffs = coeffs
        
        # Move coefficients into storage
        cls.coefficients = {}
        cls.coefficients["a"] = coeffs[0]
        if cls.N_levels>1:
            for i in range( cls.N_levels ):
                cls.coefficients[f"d_l{i}"] = coeffs[i+1]
        else:
            cls.coefficients["d"] = coeffs[1]
        """
            Note that we are storing the coefficients in the following format:

        For single level:
            coefficients["a"] -> The approximation coefficients
            coefficients["d"] -> The detail coefficients

        For multi-level:
            coefficients["a"] -> The approximation coefficients
            coefficients["d_l<level index>"] -> The detail coefficients

            Note that the level index increases with more detail.

        """

    def derivatives(cls, storage_level=0, verbosity=0, nan_value=0 ):
        """
            This method calculates the spatial derivatives via the DWT projection method

        Args:
            storage_level (int, optional):  The amount of things to store in the object.

            verbosity (int, optional):   The amount of information to print to the terminal.

            nan_value (float, optional): The value to use for NaN values. Defaults to 0.

        """
        import pywt

        # Initialize the list that stores the derivatives
        cls.derivatives = []

        #=============================================================
        #
        #   Calculate the 1st Spatial Derivative
        #
        #=============================================================

        # Initialize storage value
        if storage_level>0:
            cls.firstDeriv_raw = {}
            cls.subgridFirstDeriv_raw = {}

        # Initialize the storage of the 1st derivative coefficients
        first_derivative = {}
        if storage_level>0:
            cls.first_derivative = {}

        for i, k in enumerate( list( cls.coefficients.keys() ) ):
            if verbosity>0:
                print(f"i={i}")
            # Pull the coefficients at the level the key describes
            coefficients_atLevel = cls.coefficients[k]
            level_i = i-1
            
            # Initialize a numpy matrix that will hold the data for the 1st derivative calculation
            deriv1_raw = np.zeros( ( 2, coefficients_atLevel.shape[0] ) )

            # Calculate the spacing between values
            dx = np.gradient( cls.DWT_domain[max(0,level_i)], edge_order=1 )

            # Calculate the first derivative for the coefficient term
            if (k=="a") or ( cls.gradient_construction.lower()=="full" ):
                if cls.spatial_order<=2:
                    if verbosity>0:
                        print(f"Coefficients at the level:\t{coefficients_atLevel.shape}")
                        print(f"DWT domain at the level {max(0,level_i)}:\t{cls.DWT_domain[max(0,level_i)].shape}")
                    #deriv1_raw[0]=np.gradient( coefficients_atLevel, cls.DWT_domain[max(0,level_i)], edge_order=cls.spatialBC_order )
                    deriv1_raw[0]=np.gradient( coefficients_atLevel, np.mean(dx[2:-2]), edge_order=cls.spatialBC_order )
                else:
                    import warnings
                    warnings.warn("Wavelet equation object does not currently support orders >2, defaulting to Central Differencing", UserWarning)
                    deriv1_raw[0]=np.gradient( coefficients_atLevel, cls.DWT_domain[max(0,level_i)], edge_order=cls.spatialBC_order )   

            # Calculate the first derivative for the subgrid term
            subgrid = np.zeros( ( len(dx), len(dx) ) )
            for jj in range( len(dx) ):
                if verbosity>0:
                    print(f"jj={jj}")
                if jj<(cls.n_padded//4) or jj>(len(dx)-cls.n_padded//4-1):
                    if verbosity>1:
                        print(f"Edge found")
                else:
                    start = max( 0, jj-len(cls.wavelet_shapes_deriv_convolution["psi'*psi"])//2 )
                    end = min( len(dx), jj+len(cls.wavelet_shapes_deriv_convolution["psi'*psi"])//2+1 )
                    if k.startswith("d"):
                        subgrid_insert = cls.wavelet_shapes_deriv_convolution["psi'*psi"]
                        #subgrid[jj,start:end] = (2**(i-1))*subgrid_insert/(dx[jj])
                        subgrid[jj,start:end] = subgrid_insert/(dx[jj])
                    elif k.startswith("a") and ( cls.gradient_construction.lower() in ["modified", "full"] ):
                        subgrid_insert = cls.wavelet_shapes_deriv_convolution["phi'*phi"]
                        #subgrid[jj,start:end] = (2**(i-1))*subgrid_insert/(dx[jj])
                        subgrid[jj,start:end] = subgrid_insert/(dx[jj])
            deriv1_raw[1] = np.matmul( subgrid, coefficients_atLevel )
            if storage_level>0:
                cls.subgridFirstDeriv_raw[k+"_1stDer"] = subgrid

            # Filter NaN's
            if not nan_value is None:
                deriv1_raw = np.nan_to_num( np.array( deriv1_raw ).astype(np.float64), nan=nan_value )

            # Calculate the first derivative
            first_derivative[k] = np.sum( np.array( deriv1_raw ), axis=0 )
            if storage_level>0:
                cls.first_derivative[k] = first_derivative[k]

            if storage_level>0:
                cls.firstDeriv_raw[k+"_1stDer"] = deriv1_raw

        cls.derivatives += [first_derivative]

        #=============================================================
        #
        #   Calculate the 2nd Spatial Derivative
        #
        #=============================================================
        #"""
        # Initialize storage value
        if storage_level>0:
            cls.secondDeriv_raw = {}
            cls.subgridSecondDeriv_raw = {}
            cls.dilationSecondDeriv_raw = {}

        # Initialize the storage of the 2nd derivative coefficients
        second_derivative = {}
        if storage_level>0:
            cls.second_derivative = {}

        for i, k in enumerate( list( cls.coefficients.keys() ) ):
            if verbosity>0:
                print(f"i={i}")
            # Pull the coefficients at the level the key describes
            coefficients_atLevel = cls.coefficients[k]
            level_i = i-1
            
            # Initialize a numpy matrix that will hold the data for the 2nd derivative calculation
            deriv2_raw = np.zeros( ( 3, coefficients_atLevel.shape[0] ) )

            # Calculate the spacing between values
            dx = np.gradient( cls.DWT_domain[max(0,level_i)], edge_order=1 )

            # Calculate the first derivative for the coefficient term
            if (k=="a") or ( cls.gradient_construction.lower()=="full" ):
                if cls.spatial_order<=2:
                    if verbosity>0:
                        print(f"Coefficients at the level:\t{coefficients_atLevel.shape}")
                        print(f"DWT domain at the level {max(0,level_i)}:\t{cls.DWT_domain[max(0,level_i)].shape}")
                    deriv2_raw[0]=np.gradient( np.gradient( coefficients_atLevel, cls.DWT_domain[max(0,level_i)], edge_order=cls.spatialBC_order ), cls.DWT_domain[max(0,level_i)], edge_order=cls.spatialBC_order )
                else:
                    import warnings
                    warnings.warn("Wavelet equation object does not currently support orders >2, defaulting to Central Differencing", UserWarning)
                    deriv2_raw[0]=np.gradient( np.gradient( coefficients_atLevel, cls.DWT_domain[max(0,level_i)], edge_order=cls.spatialBC_order ), cls.DWT_domain[max(0,level_i)], edge_order=cls.spatialBC_order )

            # Calculate the second derivative for the dilation term
            subgrid = np.zeros( ( len(dx), len(dx) ) )
            for jj in range( len(dx) ):
                if verbosity>0:
                    print(f"jj={jj}")
                if jj<(cls.n_padded//4) or jj>(len(dx)-cls.n_padded//4-1):
                    if verbosity>1:
                        print(f"Edge found")
                else:
                    start = max( 0, jj-len(cls.wavelet_shapes_deriv_convolution["psi'*psi"])//2 )
                    end = min( len(dx), jj+len(cls.wavelet_shapes_deriv_convolution["psi'*psi"])//2+1 )
                    if k.startswith("d") and ( cls.gradient_construction.lower() in ["full"] ):
                        subgrid_insert = cls.wavelet_shapes_deriv_convolution["psi'*psi"]
                        subgrid[jj,start:end] = subgrid_insert/(dx[jj]*dx[jj])
                    elif k.startswith("a") and ( cls.gradient_construction.lower() in ["modified", "full"] ):
                        subgrid_insert = cls.wavelet_shapes_deriv_convolution["phi'*phi"]
                        subgrid[jj,start:end] = subgrid_insert/(dx[jj]*dx[jj])
            deriv2_raw[1] = np.matmul( subgrid, np.gradient( coefficients_atLevel, cls.DWT_domain[max(0,level_i)], edge_order=cls.spatialBC_order ) )
            if storage_level>0:
                cls.dilationSecondDeriv_raw[k+"_2ndDer"] = subgrid

            # Calculate the second derivative for the subgrid term
            subgrid = np.zeros( ( len(dx), len(dx) ) )
            for jj in range( len(dx) ):
                if verbosity>0:
                    print(f"jj={jj}")
                if jj<(cls.n_padded//4) or jj>(len(dx)-cls.n_padded//4-1):
                    if verbosity>1:
                        print(f"Edge found")
                else:
                    start = max( 0, jj-len(cls.wavelet_shapes_deriv_convolution["psi'*psi"])//2 )
                    end = min( len(dx), jj+len(cls.wavelet_shapes_deriv_convolution["psi'*psi"])//2+1 )
                    if k.startswith("d"):
                        subgrid_insert = cls.wavelet_shapes_deriv_convolution["psi''*psi"]
                        subgrid[jj,start:end] = subgrid_insert/(dx[jj])
                    elif k.startswith("a") and ( cls.gradient_construction.lower() in ["modified", "full"] ):
                        subgrid_insert = cls.wavelet_shapes_deriv_convolution["phi''*phi"]
                        subgrid[jj,start:end] = subgrid_insert/(dx[jj])
            deriv2_raw[-1] = np.matmul( subgrid, coefficients_atLevel )
            if storage_level>0:
                cls.subgridSecondDeriv_raw[k+"_2ndDer"] = subgrid

            # Filter NaN's
            if not nan_value is None:
                deriv2_raw = np.nan_to_num( np.array( deriv2_raw ), nan=nan_value )

            # Calculate the second derivative
            second_derivative[k] = np.sum( np.array( deriv2_raw ), axis=0 )
            if storage_level>0:
                cls.second_derivative[k] = second_derivative[k]

            if storage_level>0:
                cls.secondDeriv_raw[k+"_2ndDer"] = deriv2_raw

        cls.derivatives += [second_derivative]

        #"""

    def boundaryConditioning(cls, u_BC=[None, None], du_BC=[None, None], d2u_BC=[None, None], derivative_order=None, storage_level=0 ):
        """
            This method alters the coefficients and allows for the boundary conditions to allow for
        boundary condition operation.

        Args:
            u_BC (list, optional):   The boundary conditions for the function u at the left and 
                                        right boundaries, in the format: [u_LHS, u_RHS]. Defaults
                                        to [None, None].

            du_BC (list, optional):  The boundary conditions for the first derivative of u at the
                                        left and right boundaries, in the format: [du_LHS, du_RHS].
                                        Defaults to [None, None].

            d2u_BC (list, optional): The boundary conditions for the second derivative of u at the
                                        left and right boundaries, in the format: [d2u_LHS, 
                                        d2u_RHS]. Defaults to [None, None].

            derivative_order (int, optional): The order of the derivative to apply the BCs to. 
                                                Defaults to None, which generates the BC operators
                                                for the highest derivative order.

        """
        # Import PyWavelets
        import pywt

        #=============================================================
        #
        #   Set data coefficients to zero at BCs
        #
        #=============================================================
        """
        for k in list( cls.coefficients.keys() ):
            cls.coefficients[k][:1] = 0
            cls.coefficients[k][-1:] = 0
        """

        #=============================================================
        #
        #   Precompute DWT BC operators
        #
        #=============================================================

        #
        # von Neumann BC operators
        #
        cls.vonNeumann_BC_operators = {}
        u_LHS = np.zeros_like( cls.u[0] ) 
        u_RHS = np.zeros_like( cls.u[0] )
        u_LHS[0] = 1.0
        u_RHS[-1] = 1.0
        if cls.N_levels==1:
            cls.vonNeumann_BC_operators["left"] = pywt.dwt( u_LHS, cls.wavelet, mode="zero" )
            cls.vonNeumann_BC_operators["right"] = pywt.dwt( u_RHS, cls.wavelet, mode="zero" )
        elif cls.N_levels>1:
            cls.vonNeumann_BC_operators["left"] = pywt.wavedec( u_LHS, cls.wavelet, mode="zero", level=cls.N_levels )
            cls.vonNeumann_BC_operators["right"] = pywt.wavedec( u_RHS, cls.wavelet, mode="zero", level=cls.N_levels )

        # Find the maximum derivative order to apply BCs to
        if derivative_order is None:
            max_derivative_order = cls.max_derivative
        elif du_BC.any() is not None:
            max_derivative_order = 1
        elif d2u_BC.any() is not None:
            max_derivative_order = 2
            
        # Calculate the BC operators for each derivative order
        grad_objs = []
        cls.dirichlet_BC_ceoffs = {}
        cls.dirichlet_BC_ceoffs["LHS"] = []
        cls.dirichlet_BC_ceoffs["RHS"] = []
        if derivative_order is None:
            for der_order in range( 1, max_derivative_order+1 ):
                gradient_obj = numericalGradient( der_order, [ np.ceil(max_derivative_order/2).astype(int), np.ceil(max_derivative_order/2).astype(int) ]  )
                grad_objs += [ gradient_obj ]

                # Calculate the BC operators
                for i, k in enumerate( list( cls.coefficients.keys() ) ):
                    BC_operator_LHS = grad_objs[-1].coeffs_LHS
                    BC_operator_RHS = grad_objs[-1].coeffs_RHS
                cls.dirichlet_BC_ceoffs["LHS"] += [ BC_operator_LHS ]
                cls.dirichlet_BC_ceoffs["RHS"] += [ BC_operator_RHS ]

        #
        # Dirichlet BC operators
        #
        cls.dirichlet_BC_operators = []
        for der_order in range( 1, max_derivative_order+1 ):
            BC_operator = {}
            BC_operator["left"] = []
            BC_operator["right"] = []

            du_LHS = np.zeros_like( cls.u[0] )
            du_RHS = np.zeros_like( cls.u[0] )
            du_LHS[:len( cls.dirichlet_BC_ceoffs["LHS"][der_order-1] )] = cls.dirichlet_BC_ceoffs["LHS"][der_order-1]
            du_RHS[-len( cls.dirichlet_BC_ceoffs["RHS"][der_order-1] ):] = cls.dirichlet_BC_ceoffs["RHS"][der_order-1]

            if cls.N_levels==1:
                BC_operator["left"] = pywt.dwt( du_LHS, cls.wavelet, mode="zero" ) 
                BC_operator["right"] = pywt.dwt( du_RHS, cls.wavelet, mode="zero" )
            elif cls.N_levels>1:
                BC_operator["left"] = pywt.wavedec( du_LHS, cls.wavelet, mode="zero", level=cls.N_levels ) 
                BC_operator["right"] = pywt.wavedec( du_RHS, cls.wavelet, mode="zero", level=cls.N_levels )

            cls.dirichlet_BC_operators += [ BC_operator ]

    def derivative_reconstruction(cls ):
        """
            This method reconstructs the spatial derivatives back to the original domain using the
        Inverse Discrete Wavelet Transform.

        """
        import pywt

        # Set up reconstructed derivatives storage
        cls.reconstructed_derivatives = []

        # Calculate the reconstructed derivatives
        for i in range( len( cls.derivatives ) ):
            derivative_atLevel = cls.derivatives[i]
            
            if cls.N_levels>1:
                coeffs = [ derivative_atLevel["a"] ]
                for i in range( cls.N_levels ):
                    coeffs += [ derivative_atLevel[f"d_l{i}"] ]
                der = pywt.waverec( coeffs, cls.wavelet, mode=cls.signal_extension )
            else:
                der = pywt.idwt( derivative_atLevel["a"], derivative_atLevel["d"], cls.wavelet, mode=cls.signal_extension )

            cls.reconstructed_derivatives += [ der ]

    def flux_reconstruction(cls, domain_reconciliation="interpolation" ):
        """
            This method produces the reconstruction of the flux term.

        """
        import pywt

        #
        # Loop through the levels to get the convective values
        #
        og_keys = list( cls.coefficients.keys() )
        cls.convective_velocity = {}
        cls.raw_convection = {}
        cls.conv_coeffs = {}
        for i, k in enumerate( og_keys ):
            #print(f"i={i}, key {k}")

            if cls.N_levels>1:
                
                cls.conv_coeffs[k] = []
                for j, kk in enumerate( og_keys ):

                    if j<i:
                        cls.conv_coeffs[k] += [cls.coefficients[kk]]
                    elif k=="a" and kk=="a":
                        cls.conv_coeffs[k] += [cls.coefficients[kk]]
                    else:
                        cls.conv_coeffs[k] += [np.zeros_like(cls.coefficients[kk])]

                cls.raw_convection[k] = pywt.waverec( cls.conv_coeffs[k], wavelet=cls.wavelet, mode=cls.signal_extension )

            else:
                
                # Reconstruct the convective velocity 
                cls.raw_convection[k] = pywt.idwt( cls.coefficients["a"], np.zeros_like( cls.coefficients["d"] ), wavelet=cls.wavelet, mode=cls.signal_extension )


            if domain_reconciliation.lower() in ["averaging", "average", "kernel"]:
                print("Averaging under construction")
        
            elif domain_reconciliation.lower() in ["interpolation", "interpolate", "interp"]:
                
                if k=="a":
                    cls.convective_velocity[k] = np.interp( cls.DWT_domain[0], cls.x_domain, cls.raw_convection[k] )
                else:
                    cls.convective_velocity[k] = np.interp( cls.DWT_domain[i-1], cls.x_domain, cls.raw_convection[k] )



        #
        # Loop through to find the convective rate
        #
        #"""
        cls.convection = {}
        for i, k in enumerate( og_keys ):

            if k=="a":
                cls.convection[k] = cls.derivatives[0][k] * cls.convective_velocity[k]
            else:
                cls.convection[k] = cls.derivatives[0][k] * cls.convective_velocity[k]
        #"""

    def resampling(cls ):
        """
            This method resamples the wavelet coefficients to allow energy transfer between levels.

        Returns:
            _type_: _description_
        """
        import pywt

        print("Under construction")

        if cls.N_levels==1:
            u_hold = pywt.idwt( cls.coefficients["a"], cls.coefficients["d"], cls.wavelet, mode=cls.signal_extension )
            coeffs = pywt.dwt( u_hold, cls.wavelet, mode=cls.signal_extension )
            cls.coefficients["a"] = coeffs[0]
            cls.coefficients["d"] = coeffs[1]
        elif cls.N_levels>1:
            u_hold = pywt.waverec( [ cls.coefficients["a"] ] + [ cls.coefficients[f"d_l{i}"] for i in range(cls.N_levels) ], cls.wavelet, mode=cls.signal_extension )
            coeffs = pywt.wavedec( u_hold, cls.wavelet, mode=cls.signal_extension, level=cls.N_levels )
            cls.coefficients["a"] = coeffs[0]
            for i in range( cls.N_levels ):
                cls.coefficients[f"d_l{i}"] = coeffs[i+1]

class burgers_DWTeqn(wavelet_eqn):
    """
        This object contains the data and methods to solve the Burgers Equatio via Discrete Wavelet
    Transform methods. Inheritance chain is:

    General PDE problem (eqn_problem) -> DWT PDE problem (wavelet_eqn) -> Burger's Equation PDE
        problem (burgers_DWTeqn)
    
    """
    def __init__(self, x, u_0, spatial_order=2, spatialBC_order=None, stepping="explicit", viscid=True, N_levels=1, wavelet="db2", signal_extension="zero" ):
        """
            Initialize the Burgers Equation problem.

        Args:
            x (float, NumPy NDArray):   The spatial domain to perform the operation over.

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

            viscid (bool, optional):    If the problem will be a viscous Burger's equation that
                                            includes the diffusion. This in turn determines the
                                            maximum derivative that will be used.

            N_levels (int): The number of levels in the DWT.

            wavelet (str):  The type of wavelet to be used in the DWT. Defaults to "db2", or Daubechies
                                with 2 vanishing moments.
        
        """
        # Set up boundary condition order
        if spatialBC_order is None:
            spatialBC_order = spatial_order
        print(f"Spatial order is {spatial_order}")

        # Initialize from eqn_problem
        if viscid:
            super().__init__(spatial_order, spatialBC_order, stepping=stepping, max_derivative=2, N_levels=N_levels, wavelet=wavelet, signal_extension=signal_extension )
        else:
            super().__init__(spatial_order, spatialBC_order, stepping=stepping, max_derivative=1, N_levels=N_levels, wavelet=wavelet, signal_extension=signal_extension )
        self.viscid=viscid

        # Precompute the waveshapes
        super().waveshape_precompute()

        # Precompute the domain
        super().domain_initialization( x )

        # Initialize the wavelet decomposition
        super().wavelet_initialization( u_0 )

    def __call__(cls, coeffs, BCs, *args):
        """
            Set of Differential equations to solve the Burgers Equation.

            The equation is set up in the following format:

        **Explicit methods**

        <du/dt>=nu<d2u/dx2>-<F(u)>

                =nu*<L>-<F>

        Where 
        - <du/dt> is the time derivative of the solution, which is the output
        - nu is the diffusion coefficient
        - <d2u/dx2> is the second spatial gradient/Laplace operator that will be projected into 
                    each wavelet basis space.
        - <F(u)> is the vector of flux in each wavelet basis space.

        Args:
            x (np.ndarray):         The spatial grid

            u (np.ndarray):         The function of the Burgers Equation projected onto the wavelet
                                    basis spaces.

            BC_x (np.ndarray):      The boundary conditions for the spatial grid

            BC_dx (np.ndarray):     The boundary conditions for the spatial derivative

        Returns:
            du_dt (np.ndarray):     The time step function of a single time step of the Burger's 
                                    equation.
        
        """

        nu=coeffs[0]

        # Calculate derivatives
        super().derivatives()

        # Calculate the flux values
        super().flux_reconstruction()

        # Pull the boundary conditions
        BC_x = BCs[0]
        BC_dx = BCs[1]
        if cls.viscid:
            BC_dx2 = BCs[2]

        # Set up L-arrays - ie: 2nd derivative
        cls.L = {}
        if cls.viscid and not nu==0:
            for k in list( cls.derivatives[1].keys() ):
                cls.L[k] = nu * cls.derivatives[1][k]

        # Set up F-arrays
        cls.F = {}
        for k in list( cls.derivatives[0].keys() ):
            cls.F[k] = cls.convection[k]

        # Sum to time derivative
        du_dt = {}
        for k in list( cls.L.keys() ):
            du_dt[k] = -cls.F[k]
            if cls.viscid and not nu==0:
                du_dt[k] += cls.L[k]

        return du_dt