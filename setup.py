from distutils.core import setup, Extension
import numpy

loader = Extension('mghloader',
                   sources = ['src/mri.c'],
                   include_dirs=[numpy.get_include()]
)

setup (name = 'IPySurfer',
       version = '1.0',
       description = 'MRI visualizer on IPython notebook',
       author="Naoki Nishida",
       author_email="domitry@gmail.com",
       ext_modules = [loader],
       packages = ["ipysurfer"],
       package_data = {"ipysurfer":["template/*.html"]}
)
