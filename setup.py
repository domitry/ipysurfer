from distutils.core import setup, Extension
import numpy

loader = Extension('mghloader',
                   sources = ['src/mri.c'],
                   include_dirs=[numpy.get_include()]
)

setup (name = 'IPySurfer',
       version = '1.0',
       description = 'MRI visualizer on IPython notebook',
       ext_modules = [loader]
)
