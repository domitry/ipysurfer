import os, re, gzip
import numpy
import mghloader

class MRI(object):
    @classmethod
    def from_mgz(cls, fname, start=0, end=0):
        """
        Initialize MRI from .mgh or .mgz file
        ==Arguments==
        fname: path to mgh/mgz file
        """
        if re.match(r"(.+).mgz", fname):
            # unzip if compressed
            fullpath = os.path.join(os.getcwd(), "tmp.mgh")
            unziped = gzip.open(fname, "rb").read()
            file = open(fullpath, "wb")
            file.write(unziped)
            file.close()
        elif re.match(r"(.+).mgh", start, end):
            fullpath = fname
        else:
            raise Exception("Cannot read file except ones named *.mgz or *.mgh")

        arr = mghloader.read(fullpath, start, end)
        return cls(arr)

    def __init__(self, arr):
        self.data = arr
        shape = arr.shape
        self.nframes = shape[0]
        self.depth = shape[1]
        self.height = shape[2]
        self.width = shape[3]

class RawMRI(MRI):
    def to_png(self, fp):
        from PIL import Image, ImageFilter
        from math import sqrt

        sq_dep = sqrt(self.depth)
        width = self.width
        height = self.height
        depth = self.depth

        arr = self.replaced.reshape((depth*height, width))
        new_arr = numpy.empty((width*sq_dep, height*sq_dep), dtype=numpy.uint8)

        for h in range(0, int(sq_dep)):
            for w in range(0, int(sq_dep)):
                new_arr[h*height : (h+1)*height, w*width : (w+1)*width] = arr[(sq_dep*h+w)*height : (sq_dep*h+w+1)*height, :]

        img = Image.fromarray(new_arr, "L")

        img.save(fp, "PNG")

    def show(self, num=0, section="z", frame_num=0):
        """
        Show the specified frame as 2D heatmap plot.
        """
        import matplotlib.pyplot as plt
        frame = self.data[frame_num]
        if section=="z":
            img = frame[num, :, :]
        elif section=="y":
            img = frame[:, num, :]
        elif section=="x":
            img = frame[:, :, num]
        else:
            raise Exception("Section should be specified by \"x\", \"y\", \"z\"")
            
        plt.imshow(img, cmap=plt.cm.gray)

class CategorizedMRI(MRI):
    @classmethod
    def from_mgz(cls, fname, label, start=0, end=0):
        mri = super(CategorizedMRI, cls).from_mgz(fname, start, end)
        mri.register_label(label)
        return mri

    def show(self, num=0):
        import matplotlib.pyplot as plt
        arr = self.replaced[num]
        plt.imshow(arr)

    def to_png(self, fp, filter=False):
        from PIL import Image, ImageFilter
        from math import sqrt

        sq_dep = sqrt(self.depth)
        width = self.width
        height = self.height
        depth = self.depth

        arr = self.replaced.reshape((depth*height, width, 3))
        new_arr = numpy.empty((width*sq_dep, height*sq_dep, 3), dtype=numpy.uint8)

        for h in range(0, int(sq_dep)):
            for w in range(0, int(sq_dep)):
                new_arr[h*height : (h+1)*height, w*width : (w+1)*width] = arr[(sq_dep*h+w)*height : (sq_dep*h+w+1)*height, :]

        img = Image.fromarray(new_arr, "RGB")

        if filter == True:
            img = img.filter(ImageFilter.SMOOTH)

        img.save(fp, "PNG")

    def register_label(self, fname, frame=0):
        txt = open(fname)
        rule = {}
        for row in txt:
            # number, name, R, G, B, A
            m = re.match(r"(\d+)\s+([0-9a-zA-Z\_\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d)", row)
            if m is not None:
                g = m.groups()
                rule[int(g[0])] = (g[1], numpy.array([int(v) for v in g[2:5]], dtype=numpy.uint8))

        arr = numpy.empty(self.data.shape[1:]+(3,), dtype=numpy.uint8)
        rule_alive = {}

        for x in range(0, self.width):
            for y in range(0, self.height):
                for z in range(0, self.depth):
                    region = self.data[frame, z, y, x]
                    arr[z, y, x] = rule[region][1]
                    rule_alive[region]= rule[region]

        self.replaced = arr
        self.label = rule_alive
