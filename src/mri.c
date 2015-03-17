#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <stdio.h>
#include <stdlib.h>

#include "mri.h"

// internal use
static PyObject* read_mgh_(const char* fname){
  FILE *fp;
  MGH_HEADER *header = (MGH_HEADER*)malloc(sizeof(MGH_HEADER));
  int nvvoxel, dtype;
  npy_intp dims[4];
  size_t err;

  void *mri;
  PyObject* obj;

  if((fp = fopen(fname, "rb")) == NULL){
    PyErr_SetString(PyExc_Exception, "File cannot be opened.");
    return NULL;
  }

  if(fread(header, sizeof(MGH_HEADER), 1, fp) < 1){
    PyErr_SetString(PyExc_Exception, "header reading failed.");
    return NULL;
  };

  nvvoxel = header->width * header->height * header->depth * header->nframes;
  dims[0] = header->nframes;
  dims[1] = header->depth;
  dims[2] = header->height;
  dims[3] = header->width;

  switch(header->type){
  case MRI_UCHAR:
    mri = malloc(sizeof(uchar)*nvvoxel);
    err = fread(mri, sizeof(uchar), nvvoxel, fp);
    dtype = NPY_INT8;
    break;
  case MRI_INT:
    mri = malloc(sizeof(int)*nvvoxel);
    err = fread(mri, sizeof(int), nvvoxel, fp);
    dtype = NPY_INT;
    break;
  case MRI_LONG:
    mri = malloc(sizeof(long)*nvvoxel);
    err = fread(mri, sizeof(long), nvvoxel, fp);
    dtype = NPY_LONG;
    break;
  case MRI_FLOAT:
    mri = malloc(sizeof(float)*nvvoxel);
    err = fread(mri, sizeof(float), nvvoxel, fp);
    dtype = NPY_FLOAT;
    break;
  case MRI_SHORT:
    mri = malloc(sizeof(short)*nvvoxel);
    err = fread(mri, sizeof(short), nvvoxel, fp);
    dtype = NPY_SHORT;
    break;
  case MRI_BITMAP:
  default:
    PyErr_SetString(PyExc_Exception, "Cannot read this type of mgh.");
    return NULL;
  }

  if(err < nvvoxel){
    PyErr_SetString(PyExc_Exception, "voxel data reading failed.");
    return NULL;
  }
  
  obj = PyArray_SimpleNewFromData(4, dims, dtype, mri);

  fclose(fp);
  return obj;
}

// Read .mgh file specified by arguments
// example:
//   read_mgh("path_to_mgh")
//
static PyObject* read_mgh(PyObject *self, PyObject *args){
  const char *fname;
  PyArg_ParseTuple(args, "s", &fname);
  return read_mgh_(fname);
}

//TODO: Accept #if #else and enable to be executed on Python3

static PyMethodDef methods[] = {
  {
    "read", read_mgh, METH_VARARGS, "read .mgh file"
  },
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmghloader(void){
  (void)Py_InitModule("mghloader", methods);
  import_array();
}
