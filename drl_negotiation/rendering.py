"""
rendering framework for negotiation and scml
"""
import os
import six
import sys

from gym import error

try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

try:
    from pyglet.gl import *
except ImportError as e:
     raise ImportError('''
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')

import math
import numpy as np

def get_display(spec):
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(f"Invalid display specification:{spec}")

class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)
        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        
        glEnable(GL_BLEND)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(2.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def add_geom(self, geom):
        self.geoms.append(geom)

    def render(self, return_rgb_array=False):
        pass

