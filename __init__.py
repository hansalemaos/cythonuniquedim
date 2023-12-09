import os
import subprocess
import sys
from cythonanyarray import get_iterarray, get_iterarray_shape, get_pointer_array
import numpy as np


def _dummyimport():
    import Cython
    import gmpy2


try:
    from .uniqgum import gmpy_dict_full_keys, gmpy_dict_int_keys, gmpy_list, unpack_numbers, unique_dims

except Exception as e:
    cstring = r"""# distutils: language=c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: binding=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

#include "gmp.h"
#include "mpc.h"
#include "mpfr.h"

from gmpy2 cimport *
cimport cython
import cython
import numpy as np
cimport numpy as np
ctypedef fused urealpic:
    cython.uchar
    cython.ushort
    cython.uint
    cython.ulong
    cython.ulonglong
   
import_gmpy2()

cpdef dict[mpz,list[Py_ssize_t]] gmpy_dict_full_keys(urealpic[:,:] viuarra, cython.ulonglong usebitshift_int ):
    cdef mpz bitmap =mpz(0)
    cdef mpz bitmaptmp = mpz(0)
    cdef mpz bitmap2 = mpz(0)
    cdef mpz bitmap2tmp = mpz(0)
    cdef mpz bitshift = mpz(1)
    cdef mpz bitshifttmp = mpz(1)
    cdef mpz usebitshift=mpz(usebitshift_int)
    cdef Py_ssize_t arrayloop=viuarra.shape[0]
    cdef Py_ssize_t viuarralen=viuarra.shape[1]
    cdef Py_ssize_t h,i
    cdef dict[mpz,list[Py_ssize_t]] dikeys = {}
    
    for h in range(arrayloop):
        for i in range(viuarralen):
            bitmap2 += (viuarra[h][i])
            bitmap2 *=  bitshift
            bitmap += bitmap2
            bitshift *= usebitshift
            bitmap2 = bitmap2tmp
        dikeys.setdefault(bitmap,[]).append(h)
        bitshift = bitshifttmp
        bitmap=bitmaptmp
    return dikeys

cpdef dict[Py_ssize_t,list[Py_ssize_t]] gmpy_dict_int_keys(urealpic[:,:] viuarra, cython.ulonglong usebitshift_int ):
    cdef dict[Py_ssize_t,list[Py_ssize_t]] mapdict = {}
    cdef list v
    cdef Py_ssize_t mapkeyindex=0
    cdef dict[mpz,list[Py_ssize_t]] dikeys= gmpy_dict_full_keys(viuarra, usebitshift_int )
    for v in dikeys.values():
        mapdict[mapkeyindex]=v
        mapkeyindex+=1
    return mapdict

cpdef list[mpz] gmpy_list(urealpic[:,:] viuarra, cython.ulonglong usebitshift_int ):
    cdef mpz bitmap =mpz(0)
    cdef mpz bitmaptmp = mpz(0)
    cdef mpz bitmap2 = mpz(0)
    cdef mpz bitmap2tmp = mpz(0)
    cdef mpz bitshift = mpz(1)
    cdef mpz bitshifttmp = mpz(1)
    cdef mpz usebitshift=mpz(usebitshift_int)
    cdef Py_ssize_t arrayloop=viuarra.shape[0]
    cdef Py_ssize_t viuarralen=viuarra.shape[1]
    cdef Py_ssize_t h,i
    cdef list[mpz] dikeys=[]
    for h in range(arrayloop):
        for i in range(viuarralen):
            bitmap2 += (viuarra[h][i])
            bitmap2 *=  bitshift
            bitmap += bitmap2
            bitshift *= usebitshift
            bitmap2 = bitmap2tmp
        dikeys.append(bitmap) 
        bitshift = bitshifttmp
        bitmap=bitmaptmp
    return dikeys


cpdef list[list[int]] unpack_numbers(list[mpz] viuarra, cython.ulonglong usebitshift_int ,Py_ssize_t size):
    cdef list[list[int]] resu = []
    cdef mpz bitshift = mpz(usebitshift_int)
    cdef mpz partval,newval
    cdef Py_ssize_t loopval = len(viuarra)
    cdef Py_ssize_t r3,h
    for r3 in range(loopval):
        newval = viuarra[r3]
        partval = mpz(0)
        resu.append([])
        for h in range((size)):
            partval = newval % bitshift
            resu[r3].append(int(partval))
            newval = newval // bitshift
    return resu

cpdef Py_ssize_t[:,:] unique_dims(Py_ssize_t[:,:]rax,cython.uchar[:] pointerdata,Py_ssize_t steps,Py_ssize_t productshape):
    cdef Py_ssize_t lenofarray=rax.shape[0]
    cdef set[mpz] seti=set()
    cdef list[Py_ssize_t[:]] nonduplicates=[]
    cdef mpz bitmap =mpz(0)
    cdef mpz bitmaptmp = mpz(0)
    cdef mpz bitmap2 = mpz(0)
    cdef mpz bitmap2tmp = mpz(0)
    cdef mpz bitshift = mpz(1)
    cdef mpz bitshifttmp = mpz(1)
    cdef mpz usebitshift=mpz(256)
    cdef Py_ssize_t multipl=pointerdata.shape[0]//productshape
    cdef Py_ssize_t newsteps=steps*multipl
    cdef Py_ssize_t  abs_index ,q,i
    for q in range(lenofarray):
        abs_index=rax[q][0]

        for i in range(newsteps):
            bitmap2 += (pointerdata[abs_index*multipl:abs_index*multipl+newsteps+1][i])
            bitmap2 *= bitshift
            bitmap += bitmap2
            bitshift *= usebitshift
            bitmap2 = bitmap2tmp

        if bitmap not in seti:
            nonduplicates.append(rax[q])
            seti.add(bitmap)

        bitshift = bitshifttmp
        bitmap = bitmaptmp
    return np.asarray(nonduplicates)


"""
    pyxfile = f"uniqgum.pyx"
    pyxfilesetup = f"uniqgumcompiled_setup.py"

    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    pyxfile_setup_complete_path = os.path.join(dirname, pyxfilesetup)

    if os.path.exists(pyxfile_complete_path):
        os.remove(pyxfile_complete_path)
    if os.path.exists(pyxfile_setup_complete_path):
        os.remove(pyxfile_setup_complete_path)
    with open(pyxfile_complete_path, mode="w", encoding="utf-8") as f:
        f.write(cstring)
    numpyincludefolder = np.get_include()
    compilefile = (
            """
	from setuptools import Extension, setup
	from Cython.Build import cythonize
	ext_modules = Extension(**{'py_limited_api': False, 'name': 'uniqgum', 'sources': ['uniqgum.pyx'], 'include_dirs': [\'"""
            + numpyincludefolder
            + """\'], 'define_macros': [], 'undef_macros': [], 'library_dirs': [], 'libraries': [], 'runtime_library_dirs': [], 'extra_objects': [], 'extra_compile_args': [], 'extra_link_args': [], 'export_symbols': [], 'swig_opts': [], 'depends': [], 'language': None, 'optional': None})

	setup(
		name='uniqgum',
		ext_modules=cythonize(ext_modules),
	)
			"""
    )
    with open(pyxfile_setup_complete_path, mode="w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [x.lstrip().replace(os.sep, "/") for x in compilefile.splitlines()]
            )
        )
    subprocess.run(
        [sys.executable, pyxfile_setup_complete_path, "build_ext", "--inplace"],
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    try:
        from .uniqgum import gmpy_dict_full_keys, gmpy_dict_int_keys, gmpy_list, unpack_numbers, unique_dims

    except Exception as fe:
        sys.stderr.write(f'{fe}')
        sys.stderr.flush()


def _group_same_rows_in_dict(a, bitshift_dtype=None, fullkeys=True, listoutput=False):
    if bitshift_dtype:
        convlist = {np.iinfo(bitshift_dtype).max + 1: bitshift_dtype}
    else:
        convlist = {
            2 ** 32: np.uint32, 2 ** 16: np.uint16, 2 ** 8: np.uint8
        }

    if not a.flags['C_CONTIGUOUS']:
        acopy = np.ascontiguousarray(a)
    else:
        acopy = a
    usedbitshift = None
    arrayviewshape = ()
    results = {}
    for co in convlist.items():
        try:
            viuarra = np.ascontiguousarray(acopy.view('V1').view(co[1]), dtype=co[1])
            usebitshift = (co[0])
            if not listoutput:
                if fullkeys:
                    results = gmpy_dict_full_keys(viuarra, usebitshift)
                else:
                    results = gmpy_dict_int_keys(viuarra, usebitshift)
            else:
                results = gmpy_list(viuarra, usebitshift)

            usedbitshift = usebitshift
            arrayviewshape = viuarra.shape
            break
        except Exception as e:
            sys.stderr.write(f'{co[1]} is too big\n')
            sys.stderr.flush()
    return results, usedbitshift, arrayviewshape


def unique_dimensions(a, last_dim, unordered=True, iterray=()):
    r"""
    Find unique dimensions in a multi-dimensional numpy array.

    Args:
        a (numpy.ndarray): The input multi-dimensional array.
        last_dim (int): The last dimension to consider when finding unique dimensions.
        unordered (bool, optional): If True, uses multiprocessing to create the index. Defaults to True.
        iterray (Any, optional): Custom iterator array. Defaults to an empty tuple.

    Returns:
        numpy.ndarray: The array containing unique dimensions based on the specified criteria.


    """
    data = np.ascontiguousarray(a)
    dtype = np.int64
    if isinstance(iterray, tuple):
        iterray = get_iterarray(data, dtype=dtype, unordered=unordered)

    rax = get_iterarray_shape(iterray, last_dim)
    steps = np.product(data.shape[last_dim - 1:])
    pointerdata = get_pointer_array(data).view('V1').view(np.uint8)

    fe = unique_dims(rax, pointerdata, steps, np.product(data.shape))
    return data[*[fe[..., x] for x in range(1, fe.shape[1])]]


def group_same_rows_as_dict(a, bitshift_dtype=None, fullkeys=True):
    r"""
    Group rows with the same values in a 2D numpy array and return the result as a dictionary.

    Args:
        a (numpy.ndarray): The input 2D array.
        bitshift_dtype (numpy.dtype, optional): The data type for bit shifting. Defaults to None (will find the fastest).
        fullkeys (bool, optional): If True, uses full keys (mpz) in the result dictionary.
                                   If False, uses integer keys. Defaults to True.

    Returns:
        Tuple[Dict[Union[int, mpz], List[int]], int, Tuple[int, int]]: A tuple containing the result
        dictionary, the bitshift value used, and the shape of the input array.


    """
    return _group_same_rows_in_dict(a, bitshift_dtype=bitshift_dtype, fullkeys=fullkeys, listoutput=False)


def compress_rows_as_list(a, bitshift_dtype=None):
    r"""
    Compress rows of a 2D numpy array into a list of integers using bit shifting.

    Args:
        a (numpy.ndarray): The input 2D array.
        bitshift_dtype (numpy.dtype, optional): The data type for bit shifting. Defaults to None (will find the fastest).

    Returns:
        Tuple[List[mpz], int, Tuple[int, int]]: A tuple containing the compressed list of integers,
        the bitshift value used, and the shape of the input array.

    """
    return _group_same_rows_in_dict(a, bitshift_dtype=bitshift_dtype, fullkeys=False, listoutput=True)


def uncompress_list(mpz_list, usedbitshift, elements_per_row, numpy_dtype=None):
    r"""
    Decompress a list of integers into a 2D numpy array.

    Args:
        mpz_list (List[mpz]): The compressed list of integers.
        usedbitshift (int): The bitshift value used during compression.
        elements_per_row (int): The number of elements per row in the original array.
        numpy_dtype (numpy.dtype, optional): The desired data type of the output array. Defaults to None.

    Returns:
        numpy.ndarray: The decompressed 2D numpy array.

    """
    unclist = unpack_numbers(mpz_list, usedbitshift, elements_per_row)
    if numpy_dtype:
        unclist = np.ascontiguousarray(np.array(unclist).view(numpy_dtype))
    return unclist
