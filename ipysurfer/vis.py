import os, re, json
import numpy
from math import sqrt

def plot(mri, label=None, config={}):
    return

def plot_raw_mri(mri, filter=False, config={}):
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
    mri.to_png(f, filter=filter)
    f.seek(0)
    png = "data:image/png;base64," + b64encode(f.read())
    size = {
        "width": int(mri.width),
        "height": int(mri.height),
        "depth": int(mri.depth),
        "frames_per_row": int(sqrt(mri.depth)),
        "frames_per_column": int(sqrt(mri.depth))
        }
    config.update({"voltex_size": size})

    label_rules = [[v[0], map(int, list(v[1]))] for k, v in mri.label.items()]

    html = template.render(**{
            "div_id": "vis" + str(uuid4()),
            "encoded_png": png,
            "label" : label_rules,
            "config": config
            })

    display(HTML(html))

def plot_categorized_mri():
    return
