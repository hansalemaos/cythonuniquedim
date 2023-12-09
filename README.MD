# Finds unique dimensions in multi-dimensional arrays, compresses and decompresses arrays ...


## pip install cythonuniquedim

### Tested against Windows / Python 3.11 / Anaconda


## Cython (and a C/C++ compiler) must be installed


This module provides functions for working with multi-dimensional arrays and performing operations
such as grouping rows with the same values, compressing and decompressing arrays, and
finding unique dimensions in multi-dimensional arrays.


```python
FUNCTIONS
    compress_rows_as_list(a, bitshift_dtype=None)
        Compress rows of a 2D numpy array into a list of integers using bit shifting.
        
        Args:
            a (numpy.ndarray): The input 2D array.
            bitshift_dtype (numpy.dtype, optional): The data type for bit shifting. Defaults to None (will find the fastest).
        
        Returns:
            Tuple[List[mpz], int, Tuple[int, int]]: A tuple containing the compressed list of integers,
            the bitshift value used, and the shape of the input array.
    
    group_same_rows_as_dict(a, bitshift_dtype=None, fullkeys=True)
        Group rows with the same values in a 2D numpy array and return the result as a dictionary.
        
        Args:
            a (numpy.ndarray): The input 2D array.
            bitshift_dtype (numpy.dtype, optional): The data type for bit shifting. Defaults to None (will find the fastest).
            fullkeys (bool, optional): If True, uses full keys (mpz) in the result dictionary.
                                       If False, uses integer keys. Defaults to True.
        
        Returns:
            Tuple[Dict[Union[int, mpz], List[int]], int, Tuple[int, int]]: A tuple containing the result
            dictionary, the bitshift value used, and the shape of the input array.
    
    uncompress_list(mpz_list, usedbitshift, elements_per_row, numpy_dtype=None)
        Decompress a list of integers into a 2D numpy array.
        
        Args:
            mpz_list (List[mpz]): The compressed list of integers.
            usedbitshift (int): The bitshift value used during compression.
            elements_per_row (int): The number of elements per row in the original array.
            numpy_dtype (numpy.dtype, optional): The desired data type of the output array. Defaults to None.
        
        Returns:
            numpy.ndarray: The decompressed 2D numpy array.
    
    unique_dimensions(a, last_dim, unordered=True, iterray=())
        Find unique dimensions in a multi-dimensional numpy array.
        
        Args:
            a (numpy.ndarray): The input multi-dimensional array.
            last_dim (int): The last dimension to consider when finding unique dimensions.
            unordered (bool, optional): If True, uses multiprocessing to create the index. Defaults to True.
            iterray (Any, optional): Custom iterator array. Defaults to an empty tuple.
        
        Returns:
            numpy.ndarray: The array containing unique dimensions based on the specified criteria.
import numpy as np

from cythonuniquedim import group_same_rows_as_dict, compress_rows_as_list, uncompress_list, unique_dimensions

shaha = (3, 5)
dtype = np.uint8
b = np.arange(np.product(shaha), dtype=dtype)
a = b.reshape(shaha)
a = np.tile(a, 3).T
acp = a.copy()
resultdict0, usedbitshift0, arrayviewshape0 = group_same_rows_as_dict(a, bitshift_dtype=None, fullkeys=True)
resultdict1, usedbitshift1, arrayviewshape1 = group_same_rows_as_dict(a, bitshift_dtype=None, fullkeys=False)
print(f'{resultdict0,usedbitshift0,arrayviewshape0=}')
print(f'{resultdict1,usedbitshift1,arrayviewshape1=}')

mpz_list100, mpz_list_usedbitshift100, arrayviewshape100 = compress_rows_as_list(a, bitshift_dtype=None, )
mpz_list111, mpz_list_usedbitshift111, arrayviewshape111 = compress_rows_as_list(a, bitshift_dtype=np.uint8, )
print(f'{mpz_list100,mpz_list_usedbitshift100,arrayviewshape100=}')
print(f'{mpz_list111,mpz_list_usedbitshift111,arrayviewshape111=}')
unclist = (
    uncompress_list(mpz_list=mpz_list111, usedbitshift=mpz_list_usedbitshift111, elements_per_row=arrayviewshape111[1],
                    numpy_dtype=np.uint32))
print(f'{unclist=}')

# resultdict0,usedbitshift0,arrayviewshape0=({mpz(656640): [0, 5, 10], mpz(722433): [1, 6, 11], mpz(788226): [2, 7, 12], mpz(854019): [3, 8, 13], mpz(919812): [4, 9, 14]}, 256, (15, 3))
# resultdict1,usedbitshift1,arrayviewshape1=({0: [0, 5, 10], 1: [1, 6, 11], 2: [2, 7, 12], 3: [3, 8, 13], 4: [4, 9, 14]}, 256, (15, 3))
# mpz_list100,mpz_list_usedbitshift100,arrayviewshape100=([mpz(656640), mpz(722433), mpz(788226), mpz(854019), mpz(919812), mpz(656640), mpz(722433), mpz(788226), mpz(854019), mpz(919812), mpz(656640), mpz(722433), mpz(788226), mpz(854019), mpz(919812)], 256, (15, 3))
# mpz_list111,mpz_list_usedbitshift111,arrayviewshape111=([mpz(656640), mpz(722433), mpz(788226), mpz(854019), mpz(919812), mpz(656640), mpz(722433), mpz(788226), mpz(854019), mpz(919812), mpz(656640), mpz(722433), mpz(788226), mpz(854019), mpz(919812)], 256, (15, 3))
# unclist=array([[ 0,  5, 10],
#        [ 1,  6, 11],
#        [ 2,  7, 12],
#        [ 3,  8, 13],
#        [ 4,  9, 14],
#        [ 0,  5, 10],
#        [ 1,  6, 11],
#        [ 2,  7, 12],
#        [ 3,  8, 13],
#        [ 4,  9, 14],
#        [ 0,  5, 10],
#        [ 1,  6, 11],
#        [ 2,  7, 12],
#        [ 3,  8, 13],
#        [ 4,  9, 14]], dtype=uint32)

shaha = (4, 3, 4, 3, 3, 2, 3)
dtype = np.float64
b = np.arange(np.product(shaha), dtype=dtype)
a = b.reshape(shaha)
a = np.tile(a, 3).T.copy()
unidims = unique_dimensions(a, last_dim=3, unordered=True)
print(f'{unidims=}')
print(f'{unidims.shape=}')

unidims1 = unique_dimensions(a, last_dim=4, unordered=True)
print(f'{unidims1=}')
print(f'{unidims1.shape=}')
unidims2 = unique_dimensions(a, last_dim=5, unordered=True)
print(f'{unidims2=}')
print(f'{unidims2.shape=}')
unidims3 = unique_dimensions(a, last_dim=6, unordered=True)
print(f'{unidims3=}')
print(f'{unidims3.shape=}')

# unidims=array([[[[[[0.000e+00, 6.480e+02, 1.296e+03, 1.944e+03],
#            [2.160e+02, 8.640e+02, 1.512e+03, 2.160e+03],
#            [4.320e+02, 1.080e+03, 1.728e+03, 2.376e+03]],
#           [[5.400e+01, 7.020e+02, 1.350e+03, 1.998e+03],
#            [2.700e+02, 9.180e+02, 1.566e+03, 2.214e+03],
#            [4.860e+02, 1.134e+03, 1.782e+03, 2.430e+03]],
#           [[1.080e+02, 7.560e+02, 1.404e+03, 2.052e+03],
#            [3.240e+02, 9.720e+02, 1.620e+03, 2.268e+03],
#            [5.400e+02, 1.188e+03, 1.836e+03, 2.484e+03]],
#           [[1.620e+02, 8.100e+02, 1.458e+03, 2.106e+03],
#            [3.780e+02, 1.026e+03, 1.674e+03, 2.322e+03],
#            [5.940e+02, 1.242e+03, 1.890e+03, 2.538e+03]]],
#          [[[1.800e+01, 6.660e+02, 1.314e+03, 1.962e+03],
#            [2.340e+02, 8.820e+02, 1.530e+03, 2.178e+03],
#            [4.500e+02, 1.098e+03, 1.746e+03, 2.394e+03]],
#           [[7.200e+01, 7.200e+02, 1.368e+03, 2.016e+03],
#            [2.880e+02, 9.360e+02, 1.584e+03, 2.232e+03],
#            [5.040e+02, 1.152e+03, 1.800e+03, 2.448e+03]],
#           [[1.260e+02, 7.740e+02, 1.422e+03, 2.070e+03],
#            [3.420e+02, 9.900e+02, 1.638e+03, 2.286e+03],
#            [5.580e+02, 1.206e+03, 1.854e+03, 2.502e+03]],
#           [[1.800e+02, 8.280e+02, 1.476e+03, 2.124e+03],
#            [3.960e+02, 1.044e+03, 1.692e+03, 2.340e+03],
#            [6.120e+02, 1.260e+03, 1.908e+03, 2.556e+03]]],
# ......
    
```