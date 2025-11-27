/*
 *
 * (c) 2023 Michael Fitzgerald (mpfitz@ucla.edu)
 *
 * This contains a C extension for Python that computes N-body gravitational
 * accelerations.
 *
 * For more reading on using numpy arrays with the Python C API, see here:
 *   https://numpy.org/doc/stable/user/c-info.html
 *
 */
#include <stdio.h>
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include <math.h>


#define G 6.67259e-8 // [cm^3/g/s^2]  Newton's gravitational constant
  
// this kind of function definition does not take keyword arguments
static PyObject* get_accel(PyObject *dummy, PyObject *args)
{
  PyObject* arg1=NULL;  // will contain first argument
  PyObject* arg2=NULL;  // will contain first argument
  PyObject* R=NULL; // 2d array for position, Nx3
  PyObject* M=NULL; // 1d array for mass, M
  PyObject* A=NULL; // 2d array for acceleration
  int nd_R;

  // unpack the arguments
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2))
    return NULL;

  // converts input objects to Numpy arrays
  R = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (R == NULL) return NULL;
  M = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if (M == NULL) goto fail;
  

  // unpack the dimensions of the input array
  nd_R = PyArray_NDIM(R); // number of dimensions
  npy_intp* dims_R = PyArray_DIMS(R); // the shape
  int N = dims_R[0]; // number of particles

  
  // creating a new array to be used for output
  A = PyArray_NewLikeArray(R, NPY_KEEPORDER, NULL, 1);

  
  // manipulate input into output
  double dx, dy, dz, r2, ir3, mj;
  for (int i=0; i<N; i++) {
    // initialize this particle's acceleration to zero
    for (int j=0; j<3; j++)
      *((npy_double*)PyArray_GETPTR2(A,i,j)) = 0.; // [cm/s^2]

    // compute pairwise accelerations
    for (int j=0; j<N; j++) {
      if (i==j) continue;
      dx = *((npy_double*)PyArray_GETPTR2(R,i,0)) - *((npy_double*)PyArray_GETPTR2(R,j,0)); // [cm]
      dy = *((npy_double*)PyArray_GETPTR2(R,i,1)) - *((npy_double*)PyArray_GETPTR2(R,j,1)); // [cm]
      dz = *((npy_double*)PyArray_GETPTR2(R,i,2)) - *((npy_double*)PyArray_GETPTR2(R,j,2)); // [cm]
      r2 = dx*dx + dy*dy + dz*dz; // [cm^2]
      ir3 = pow(r2,-1.5); // [cm^-3]
      mj = *((npy_double*)PyArray_GETPTR1(M,j));

      // accumulate accelerations for particle i
      *((npy_double*)PyArray_GETPTR2(A,i,0)) -= G * mj * dx * ir3; // [cm/s^2]
      *((npy_double*)PyArray_GETPTR2(A,i,1)) -= G * mj * dy * ir3; // [cm/s^2]
      *((npy_double*)PyArray_GETPTR2(A,i,2)) -= G * mj * dz * ir3; // [cm/s^2]
    }
  }

  
  // cleanup
  Py_DECREF(M); // we increased the reference count when we called PyArray_FROM_OTF. so decrease
  Py_DECREF(R); // we increased the reference count when we called PyArray_FROM_OTF. so decrease
  return A;

 fail:
  // cleanup
  Py_DECREF(M); // we increased the reference count when we called PyArray_FROM_OTF. so decrease
  Py_DECREF(R); // we increased the reference count when we called PyArray_FROM_OTF. so decrease
  return NULL;
}


// below are items required for defining the C-extension module in Python

static PyMethodDef accel_methods[] = {
  {
    "get_accel", get_accel, METH_VARARGS,
    "compute gravitational acceleration",
  },
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef accel_definition = {
  PyModuleDef_HEAD_INIT,
  "_accel",
  "A Python module that interfaces to N-body acceleration C code.",
  -1,
  accel_methods
};


PyMODINIT_FUNC PyInit__accel(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&accel_definition);
}
