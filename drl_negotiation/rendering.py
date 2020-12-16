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

RAD2DEG = 57.29577951308232

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
    
    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
                translation = (-left*scalex, -bottom*scaley),
                scale = (scalex, scaley)
                )
    def render(self, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        self.transform.disable()
        
        # rgb mode 
        arr = None
        if return_rgb_array:
            pass
        self.window.flip()
        return arr

class Attr:
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)

class Label(Attr):
    def __init__(self, text='Test', font_size=10, pos=(0, 0)):
        self.label = pyglet.text.Label(text,
                font_name = "Times New Roman",
                color=(123, 125, 0, 200),
                font_size = font_size,
                x = pos[0], y=pos[1],
                )

    def enable(self):
        print("label enable")
        print(self.label.text)
        self.label.draw()

    def set_position(self, x, y, text):
        #print(f"label set position {x}, {y}, {text}")
        self.label.x = x
        self.label.y = y
        self.label.text = text

class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
       
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self._render()
        for attr in self.attrs:
            attr.disable()

    def _render(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def _render(self):
        if len(self.v) == 4: glBegin(GL_QUADS)
        elif len(self.v) >4: glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0) # draw each vertex
        glEnd()

        # color
        color = (self._color.vec4[0]*0.5, self._color.vec4[1]*0.5, self._color.vec4[2]*0.5, self._color.vec4[3]*0.5)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)
        glEnd()

def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def generate_pos(radius=10, res=30):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    return points

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


