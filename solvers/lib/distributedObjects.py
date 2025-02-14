"""

**distributedObjects.py**

@Author:    Matthew Holland
@Date:      2025-02-14
@Version:   0.0
@Contact:   matthew.holland@my.utsa.edu

    This module contains all the objects that are distributed in the solver portion of the program.

Version     Date            Author              Notes

0.0         2025-02-14      Matthew Holland     Initial version of the file, imported objects from
                                                    ME 5653 repository, available at: https://github.com/mhland10/me5653_CFD_repo
                
"""

#==================================================================================================
#
# Importing Required Modules
#
#==================================================================================================

import numpy as np
import scipy.special as spsp
import scipy.sparse as spsr
from numba import njit, prange, jit

#==================================================================================================
#
# Spatial Gradients
#
#==================================================================================================

class numericalGradient:

    def __init__( self , derivativeOrder , template ):
        """

        This object contains the data pertaining to a numerical gradient

        Args:
            derivativeOrder (int):  The order of the derivative that will be used.

            template ((int)):       The terms in the template that will be used for the
                                        gradient. This will be a tuple of (2x) entries.
                                        The first entry is the number of entries on the 
                                        LHS of the reference point. The second/last 
                                        entry is the number of entries on the RHS of
                                        the reference point.

        Attributes:

            derivativeOrder <-  Args of the same

            template        <-  Args of the same

            coeffs [float]: The coefficients of the numerical gradient according to the
                                template that was put in the object.

        """

        factorial_numba(1)

        if len( template ) > 2:
            raise ValueError( "Too many values in \"template\". Must be 2 entries." )
        elif len( template ) < 2:
            raise ValueError( "Too few values in \"template\". Must be 2 entries." )

        self.derivativeOrder = derivativeOrder
        self.template = template

        self.coeffs = gradientCoefficients( self.derivativeOrder , self.template[0] , self.template[1] , self.derivativeOrder )
        self.coeffs_LHS = gradientCoefficients( self.derivativeOrder , 0 , self.template[0] + self.template[1] , self.derivativeOrder )
        self.coeffs_RHS = gradientCoefficients( self.derivativeOrder , self.template[0] + self.template[1] , 0 , self.derivativeOrder )

    def formMatrix( cls , nPoints , acceleration = None ):
        """

        Form the matrix that calculates the gradient defined by the object. Will follow
            the format:

        [A]<u>=<u^(f)>, where f is the order of the derivative, representing such.

        It will store the [A] is the diagonal sparse format provided by SciPy.sparse

        Args:
            nPoints (int):  The number of points in the full mesh.

            accelerateion (str , optional):    The acceleration method to improve the performance of calculating the
                                        matrix. The valid options are:

                                    - *None :    No acceleration

        Attributes:
            gradientMatrix <Scipy DIA Sparse>[float]:   The matrix to find the gradients.

        
        """

        #
        # Place the data into a CSR matrix
        #
        row = []
        col = []
        data = []
        for j in range( nPoints ):
            #print("j:\t{x}".format(x=j))
            row_array = np.zeros( nPoints )
            if j < cls.template[0]:
                row_array[j:(j+len(cls.coeffs_LHS))] = cls.coeffs_LHS
            elif j >= nPoints - cls.template[0]:
                row_array[(j-len(cls.coeffs_RHS)+1):(j+1)] = cls.coeffs_RHS
            else:
                row_array[(j-cls.template[0]):(j+cls.template[1]+1)] = cls.coeffs
            #print("\trow array:\t"+str(row_array))

            row_cols_array = np.nonzero( row_array )[0]
            row_rows_array = np.asarray( [j] * len( row_cols_array ) , dtype = np.int64 )
            row_data_array = row_array[row_cols_array]
            #print( "\tColumns of non-zero:\t"+str(row_cols_array))
            #print( "\tData of non-zero:\t"+str(row_data_array))

            row += list( row_rows_array )
            col += list( row_cols_array )
            data += list( row_data_array )

        cls_data = np.asarray( data )
        cls_row = np.asarray( row , dtype = np.int64 )
        cls_col = np.asarray( col , dtype = np.int64 )
        #print("\nFinal Data:\t"+str(cls_data))
        #print("Final Rows:\t"+str(cls_row))
        #print("Final Columns:\t"+str(cls_col))

        gradientMatrix_csr = spsr.csr_matrix( ( cls_data , ( cls_row , cls_col ) ) , shape = ( nPoints , nPoints ) )

        #
        # Transfer data to DIA matrix
        #
        cls.gradientMatrix = gradientMatrix_csr.todia()

    def gradientCalc( cls , x , f_x , method = "native" ):
        """

        This method calculates the gradient associated with the discrete values entered into the method.

        Args:
            x [float]:      The discrete values in the domain to calculate the derivative over.

            f_x [float]:    The discrete values in the range to calculate the derivative over.
            
            method (str, optional):     The method of how the gradient will be calculated. The valid options
                                            are:

                                        - *"native" :   A simple matrix multiplication will be used.

                                        - "loop" :  Loop through the rows. Will transfer the matrix to CSR
                                                        to index through rows.

                                        Not case sensitive.

        Returns:
            gradient [float]:   The gradient of the function that was input to the method.

        """

        if len( f_x ) != len( x ):
            raise ValueError( "Lengths of input discrete arrays are not the same." )

        gradient = np.zeros( np.shape( f_x ) )
        dx = np.mean( np.gradient( x ) )
        cls.formMatrix( len( f_x ) )

        if method.lower()=='loop':

            for i , x_i in enumerate( x ):
                #print("i:\t{x}".format(x=i))
                csr_gradient = cls.gradientMatrix.tocsr()
                row = csr_gradient.getrow(i)
                #print("\tRow:\t"+str(row))
                #print("\tf(x):\t"+str(f_x))
                top = row * f_x
                #print("\tTop Portion:\t"+str(top))
                #print("\tdx:\t"+str(dx))
                gradient[i] = top / dx

        elif method.lower()=='native':

            gradient = cls.gradientMatrix.dot( f_x ) / dx

        else:

            raise ValueError( "Invalid method selected" )
        
        return gradient
    