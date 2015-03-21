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

    def to_png(self, fp, resize=True, frame=0):
        """
        Save MRI image as png
        """
        from PIL import Image

        if resize:
            arr = self.data[frame][::2, :, :].reshape([(self.depth/2)*self.height, self.width])
            image = Image.fromarray(arr, "L").resize((self.width/2, (self.height*self.depth)/4))
            image.save(fp, "PNG")
            return {
                "width": int(self.width/2),
                "height": int(self.height/2),
                "depth": int(self.depth/2)
            }
        else:
            arr = self.data[frame].reshape([self.depth*self.height, self.width])
            image = Image.fromarray(arr, "L")
            image.save(fp, "PNG")
            return {
                "width": int(self.width),
                "height": int(self.height),
                "depth": int(self.depth)
            }

    def plot(self, config={}):
        """
        Plot MRI Image using volume rendering.
        ===Arguments===
        config: dict
        """
        from jinja2 import Template
        from IPython.core.display import display, HTML
        from tempfile import TemporaryFile
        from base64 import b64encode
        from uuid import uuid4

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "template/vis.html"))
        template = Template(open(path).read())

        # encode png as base64
        f = TemporaryFile("r+b")
        size = self.to_png(f)
        f.seek(0)
        png = "data:image/png;base64," + b64encode(f.read())
        config.update({"voltex_size": size})

        html = template.render(**{
            "div_id": "vis" + str(uuid4()),
            "encoded_png": png,
            "config": config
        })

        display(HTML(html))
