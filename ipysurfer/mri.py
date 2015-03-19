import os, re, gzip
import numpy
import mghloader

class MRI():
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

    def show(self, frame_num=0, depth=None):
        """
        Show the specified frame as 2D heatmap plot.
        """
        import matplotlib.pyplot as plt
        frame = self.data[frame_num]
        if depth is None:
            img = frame.reshape([self.depth*self.height, self.width])
        else:
            img = frame[depth]
        plt.imshow(img, cmap=plt.cm.gray)

    def plot(self, config):
        """
        Plot MRI Image using volume rendering.
        ===Arguments===
        config: dict
        """
        from jinja2 import Template
        from IPython.core.display import display, HTML

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates/vis.html"))
        template = Template(open(path).read())

        html = template.render(**{
            "div_id": "vis" + str(uuid.uuid4())
        })

        display(HTML(html))
