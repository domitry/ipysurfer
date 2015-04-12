import os, re, json
import numpy
from math import sqrt
from .mri import MRI

class Widget(object):
    def __init__(self, fname, categorized):
        from IPython.core.display import display, Javascript
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "template/init_widget.js"))
        js = open(path).read()
        display(Javascript(js))

        self.fname = fname
        self.categorized = categorized
        self.seek = 0

    def plot(self, config={}):
        from jinja2 import Template
        from IPython.core.display import display, HTML
        from IPython.kernel.comm import Comm
        from tempfile import TemporaryFile
        from base64 import b64encode
        from uuid import uuid4

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "template/widget.html"))
        template = Template(open(path).read())

        # encode png as base64
        f = TemporaryFile("r+b")
        shape = self.categorized.to_png(f, filter=False)
        f.seek(0)
        png = "data:image/png;base64," + b64encode(f.read())
        depth, height, width = self.categorized.data.shape[1:4]
        size = {
            "width": int(width),
            "height": int(height),
            "depth": int(depth),
            "frames_per_row": int(shape[1]/width),
            "frames_per_column": int(shape[0]/height)
        }
        config.update({"voltex_size": size})

        label_rules = [[v[0], map(int, list(v[1]))] for k, v in self.categorized.label.items()]

        display(HTML(template.render(**{
            "div_id": "vis" + str(uuid4()),
            "encoded_png": png,
            "label" : label_rules,
            "config": config
        })))

        comm = Comm("IPysurfer")

        def reciever(msg):
            title = msg['content'][u'data'][u'title']
            if title == "seek":
                num = msg['content'][u'data'][u'num']
                self.seek = num
            elif title == "next":
                num = msg['content'][u'data'][u'num']
                mri = MRI.from_mgz(self.fname, start=self.seek, end=self.seek+num-1)
                comm.send({
                    "array": [mri.fold(i) for i in range(0, num)]
                })
                self.seek += num

        comm.on_msg(reciever);
