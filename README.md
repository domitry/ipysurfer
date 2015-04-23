# IPysurfer
A 3D/2D visualizer of fMRI images on IPython notebook

![](http://i.gyazo.com/ed2e81c7e54e17728969a5a45400200e.png)

## Installation
```
python setup.py install
```

## Usage
```python
from ipysurfer import mri
m = mri.from_mgz("path_to_mgz")
m.show(100) # show 100th slice
m.plot() # show 3D brain
```

## Examples:
* [sample1](http://nbviewer.ipython.org/urls/dl.dropboxusercontent.com/u/47978121/webgl/RawMRI.ipynb)
* [sample2](http://nbviewer.ipython.org/urls/dl.dropboxusercontent.com/u/47978121/webgl/Categorized_MRI.ipynb)

## License
This software is distributed under the MIT License.

IPysurfer contains some code from [WebGL-Volumetric](https://github.com/nopjia/WebGL-Volumetric), an wonderful WebGL-based visualizer created by Nop Jiarathanakul.
