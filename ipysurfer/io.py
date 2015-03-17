import os
import numpy
import mghloader


"""
return value: numpy.array
dim: [nframes, depth, height, width]
"""
def read(self, fname):
    # unzip if compressed
    if re.match(r".mgz", fname):
        # unzip and save tmp file
        fullpath = os.path.join(os.getcwd(), "tmp.mgh")
        ziped = open(fname, "rb").read()
        unziped = zlib.decompress(ziped)
        file = open(fullpath, "wb")
    elif:
        fullpath = fname
    else:
        raise "Cannot read file except *.mgz or *.mgh"

    return mghloader.read(fullpath)
