#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "mri.h"

// num: byte num
void swap_byte(void* val, int num){
  int i;
  unsigned char *p = (unsigned char *)val;
  for(i=0;i<num/2;i++){
    unsigned char a = *(p+i);
    *(p+i) = *(p+(num-i-1));
    *(p+(num-i-1)) = a;
  }
}

// internal use
static PyObject* read_mgh_(const char* fname){
  FILE *fp;
  MGH_HEADER *header = (MGH_HEADER*)malloc(sizeof(MGH_HEADER));
  int dtype, nframes, depth, height, width, area;
  int f, z, i;
  npy_intp dims[4];
  size_t size;

  bool big_end = false;
  uchar *mri;
  PyObject* obj;

  if((fp = fopen(fname, "rb")) == NULL){
    PyErr_SetString(PyExc_Exception, "File cannot be opened.");
    return NULL;
  }

  if(fread(header, sizeof(MGH_HEADER), 1, fp) < 1){
    PyErr_SetString(PyExc_Exception, "header reading failed.");
    return NULL;
  };

  if(header->version != 1){
    big_end = true;
    swap_byte(&(header->type), 4);
    swap_byte(&(header->nframes), 4);
    swap_byte(&(header->depth), 4);
    swap_byte(&(header->height), 4);
    swap_byte(&(header->width), 4);
  }

  nframes = header->nframes;
  depth = header->depth;
  height = header->height;
  width = header->width;

  printf("%d", header->type);

  switch(header->type){
  case MRI_UCHAR:
    size = sizeof(uchar);
    dtype = NPY_INT8;
    break;
  case MRI_INT:
    size = sizeof(int);
    dtype = NPY_INT;
    break;
  case MRI_LONG:
    size = sizeof(long);
    dtype = NPY_LONG;
    break;
  case MRI_FLOAT:
    size = sizeof(float);
    dtype = NPY_FLOAT;
    break;
  case MRI_SHORT:
    size = sizeof(short);
    dtype = NPY_SHORT;
    break;
  case MRI_BITMAP:
  default:
    PyErr_SetString(PyExc_Exception, "Cannot read this type of mgh.");
    return NULL;
  }

  mri = (uchar *)malloc(size*nframes*depth*width*height);
  area = width*height;

  for(f=0; f<nframes; f++){
    for(z=0; z<depth; z++){
      uchar *start = mri + f*(depth*area*size) + z*area*size;
      if(fread(start, size, area, fp) < area){
        PyErr_SetString(PyExc_Exception, "voxel data reading failed.");
        return NULL;
      }
      if(big_end)
        for(i=0; i<area; i++)
          swap_byte((start + size*i), size);
    }
  }

  dims[0] = nframes;
  dims[1] = depth;
  dims[3] = height;
  dims[2] = width;
  
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
