"""
DISTRIBUTED FUNCTIONS

Author:     Matthew Holland

This library contains functions that facilitate the rest of the code for the ME-5653 Fall 2024
    class.

"""

import numpy as np
import scipy.special as spsp
from numba import njit, prange, jit

###################################################################################################
#
# Various Math Functions
#
###################################################################################################

@njit( nogil = True , parallel = True , cache = True )
def factorial_array_numba( x ):
    y = np.zeros( np.shape( x ) )

    x_flatten = x.flatten()
    y_flatten = y.flatten()
    for i in prange( len( y_flatten ) ):
        if x_flatten[i] > 0:
            c = np.arange( 1 , x_flatten[i] )
            y_flatten[i] = np.prod( c )
        else:
            y_flatten[i] = 0
    
    y = y_flatten.reshape( np.shape(y) )

    return y

def factorial_array( x ):
    y = np.zeros( np.shape( x ) )

    x_flatten = x.flatten()
    y_flatten = y.flatten()
    for i in prange( len( y_flatten ) ):
        if x_flatten[i] > 0:
            c = np.arange( 1 , x_flatten[i] )
            y_flatten[i] = np.prod( c )
        else:
            y_flatten[i] = 0
    
    y = y_flatten.reshape( np.shape(y) )

    return y

@njit( nogil = True , cache = True )
def factorial_numba( x ):
    c = np.arange( 1 , x + 1 )
    y = np.prod( c )

    return y

def factorial( x ):
    c = np.arange( 1 , x + 1 )
    y = np.prod( c )

    return y



###################################################################################################
#
# September 2024 Functions
#
###################################################################################################

def gradientCoefficients( nOrderDerivative , negSidePoints , posSidePoints , nOrderAccuracy ):
    """
    This function calculates the coefficients for a gradient calculation for a given set of
        conditions and template.

    Args:
        nOrderDerivative <int>:     The order of the derivative that will be calculated.
        
        negSidePoints <int>:    The number of points from the reference points on the LHS or
                                    approaching negative infinity.

        posSidePoints <int>:    The number of points form the reference points on the RHS or 
                                    approaching positive infinity. 

        nOrderAccuracy <int>:   The 

    Returns:
        coeffs [float]: The array of the coefficients to the function values for the gradient that
                            is calculated.
    
    """

    n_template = negSidePoints + posSidePoints + 1
    # Calcualte the number of points in the template
    if nOrderAccuracy > n_template+1 :
        raise ValueError( "Too few points input for the order of derivative" )
    #print(  "There are {x} points in the template: {y} on the LHS and {z} on the RHS".format( x = n_template , y = negSidePoints , z = posSidePoints )   )

    points = [ x for x in range( -negSidePoints , posSidePoints + 1 ) ]
    nOrderAccuracy = len( points )
    #print( "With points:\t" + str( points ) )
    
    taylor_series_coeffs = np.ones( ( nOrderAccuracy ,) + ( nOrderAccuracy ,) )
    # Generate a matrix that contains the coefficients
    for i in range( nOrderAccuracy ):
        for j in range( nOrderAccuracy ):
            p = points[j]
            #print( "For i={x} and j={y}".format( x = i , y = j ) )
            #print( "\tp is "+str(p) )
            #c = ( p ** i ) / np.max( [ spsp.factorial( i ) , 1 ] )
            fracs = np.asarray( [ factorial( i ) , 1 ] ).max()
            #print( "\tfactorial is " + str( fracs ) )
            c = ( p ** i ) / fracs
            #print( "\tThe coefficient is {x}".format( x = c ) )
            taylor_series_coeffs[i,j] = c
        print(" ")
    # and fill out this matrix
    #print( "\nTaylor series coefficients are:\n"+str( taylor_series_coeffs ) )

    b = np.zeros( nOrderAccuracy )
    b[nOrderDerivative] = 1
    #print("b vector:\t"+str(b))
    coeffs = np.linalg.solve( taylor_series_coeffs , b )
    # Calculate the coefficients from the Taylor series coefficients
            
    return coeffs

@jit( parallel = True , boundscheck = True )
def gradientCoefficients_numba( nOrderDerivative , negSidePoints , posSidePoints , nOrderAccuracy ):
    """
    This function calculates the coefficients for a gradient calculation for a given set of
        conditions and template.

    Note: this is still in progress, I don't know why it doesnt work yet.

    Args:
        nOrderDerivative <int>:     The order of the derivative that will be calculated.
        
        negSidePoints <int>:    The number of points from the reference points on the LHS or
                                    approaching negative infinity.

        posSidePoints <int>:    The number of points form the reference points on the RHS or 
                                    approaching positive infinity. 

        nOrderAccuracy <int>:   The 

    Returns:
        coeffs [float]: The array of the coefficients to the function values for the gradient that
                            is calculated.
    
    """

    n_template = negSidePoints + posSidePoints
    # Calcualte the number of points in the template

    if negSidePoints == posSidePoints:
        points = [ x for x in range( -negSidePoints , posSidePoints + 1 ) if x != 0 ]
        nOrderAccuracy = len( points )
    else:
        points = [ x for x in range( -negSidePoints , posSidePoints + 1 ) ]
        nOrderAccuracy = len( points )
    
    taylor_series_coeffs = np.ones( ( nOrderAccuracy ,) + ( nOrderAccuracy ,) )
    # Generate a matrix that contains the coefficients
    for i in prange( nOrderAccuracy ):
        for j in prange( nOrderAccuracy ):
            p = points[j]
            c = ( p ** i ) / factorial( i )
            taylor_series_coeffs[i,j] = c
    # and fill out this matrix

    b = np.zeros( nOrderAccuracy )
    b[nOrderDerivative] = 1
    coeffs = np.linalg.solve( taylor_series_coeffs , b )
    # Calculate the coefficients from the Taylor series coefficients
            
    return coeffs

#def sparseArrayDefine_CSR( nPoints , template)