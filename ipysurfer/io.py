import os, re, gzip
import numpy
import mghloader

"""
return value: numpy.array
dim: [nframes, depth, height, width]
"""
def read(fname):
    if re.match(r"(.+).mgz", fname):
        # unzip if compressed
        fullpath = os.path.join(os.getcwd(), "tmp.mgh")
        unziped = gzip.open(fname, "rb").read()
        file = open(fullpath, "wb")
        file.write(unziped)
        file.close()
    elif re.match(r".mgh", fname):
        fullpath = fname
    else:
        raise Exception("Cannot read file except ones named *.mgz or *.mgh")

    return mghloader.read(fullpath)
