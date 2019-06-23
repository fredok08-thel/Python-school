# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:38:31 2019

@author: jazzn
"""
import math
import numpy as np
import plotly.graph_objs as go
from plotly import figure_factory as ff
from plotly.offline import  iplot
sin = np.sin
cos = np.cos
tan = np.cos
pi=np.pi

class Vector(object):
    def __init__(self, *args):
        """ Create a vector, example: v = Vector(1,2) """
        if len(args)==0: 
            self.values = (0,0)
        else: 
            self.values = args
        if len(args)>1:
            self.x=args[0]
            self.y=args[1]
        self.name='pas de nom'
        
    def norm(self):
        """ Returns the norm (length, magnitude) of the vector """
        return math.sqrt(sum( comp**2 for comp in self ))
        
    def argument(self):
        """ Returns the argument of the vector, the angle clockwise from +y."""
        arg_in_rad = math.acos(Vector(0,1)*self/self.norm())
        arg_in_deg = math.degrees(arg_in_rad)
        if self.values[0]<0: return 360 - arg_in_deg
        else: return arg_in_deg
    
        
    
    def inner(self, other):
        """ Returns the dot product (inner product) of self and other vector
        """
        return sum(a * b for a, b in zip(self, other))
    
    def __mul__(self, other):
        """ Returns the dot product of self and other if multiplied
            by another Vector.  If multiplied by an int or float,
            multiplies each component by other.
        """
        if type(other) == type(self):
            return self.inner(other)
        elif type(other) == type(1) or type(other) == type(1.0):
            product = tuple( a * other for a in self )
            resul = Vector(*product)
            resul.name=f'{other}x{self.name}'
            return resul
    
    def __rmul__(self, other):
        """ Called if 4*self for instance """
        return self.__mul__(other)
            
    def __div__(self, other):
        if type(other) == type(1) or type(other) == type(1.0):
            divided = tuple( a / other for a in self )
            return Vector(*divided)
    
    def __add__(self, other):
        """ Returns the vector addition of self and other """
        added = tuple( a + b for a, b in zip(self, other) )
        resul=Vector(*added)
        resul.name=f'{self.name}+{other.name}'
        return resul
    
    def __sub__(self, other):
        """ Returns the vector difference of self and other """
        subbed = tuple( a - b for a, b in zip(self, other) )
        resul=Vector(*subbed)
        resul.name=f'{self.name}-{other.name}'
        return resul
    
    def __iter__(self):
        return self.values.__iter__()
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, key):
        return self.values[key]
        
    def __repr__(self):
        return f'Vecteur {self.name} / coordonnées :{self.values}'

def trace_vector(vect,option='origin',scalvect=1,titre="sans titre"):
    if isinstance(vect,Vector):
        vect=[vect]
    uxL=[v[0] for v in vect]
    uyL=[v[1] for v in vect]
    xL=[0]*len(vect)
    yL=[0]*len(vect)
    if option=='a_la_suite':
        for i,v in enumerate(vect):
            if i!=0:
                xL[i]=scalvect*uxL[i-1]+xL[i-1]
                yL[i]=scalvect*uyL[i-1]+yL[i-1]
    quiver_fig = ff.create_quiver(xL,yL, uxL, uyL,
               scale=scalvect,
               arrow_scale=.05, # Sets arrow scale
               name='vecteur',
                angle=np.pi/12,
                line=dict(width=3))
    for x,y,v in zip(xL,yL,vect):
        data=go.Scatter(x=np.array(x),y=np.array(y),mode='markers',marker=dict(size=8,color='grey'),name=f'{v.name} norme={v.norm():.2f} N')
        quiver_fig.add_trace(data)
    layout= go.Layout(title= titre,xaxis= dict(constrain='domain',title= 'x'),yaxis=dict(scaleanchor='x',title= 'y'))
    quiver_fig['layout'].update(layout)
    return iplot(quiver_fig, filename='vecteur')

def trace3d(*f):
    a1=a2=np.arange(10,70,1)
    a2t = a2[:, np.newaxis]
    trace=[]
    for fnct in f:
        z=fnct(a1, a2t)
        trace.append(go.Surface(z=z,x=a1,y=a2))
    data = go.Data(trace)
    axis = dict(
    showbackground=True, # (!) show axis background
    backgroundcolor="rgb(204, 204, 204)", # set background color to grey
    gridcolor="rgb(255, 255, 255)",       # set grid line color
    zerolinecolor="rgb(255, 255, 255)",   # set zero grid line color
    )
    layout = go.Layout(scene=go.Scene(xaxis=go.XAxis(axis,title='alpha1'), # set x-axis style
        yaxis=go.YAxis(axis,title='alpha2'), # set y-axis style
        zaxis=go.ZAxis(axis,title='Force')  # set z-axis style
                )
                    )
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig, filename='s8_surface')

def radian(x):
    return x*pi/180

def degre(x):
    return x*180/pi

def trace_parametrique(func,x,y,param='x',nom_param='nom du parametre',xlabel='alpha1',ylabel='force',titre='il faut mettre un titre'):
    """
    f est une fonction de 2 variables f(x,y)
    """
    if len(y)>15:
        print(f'la liste {y} est trop grande max 15 valeurs')
        return None
    if callable(func):
        func=[func]               
    data=[]
    for elem_f in func:
        for elem_y in y:
            if param=='x':
                data.append(go.Scatter(x=x,y=elem_f(elem_y,x),mode='lines',name=f'{nom_param}={elem_y}'))
            elif param=='y':
                data.append(go.Scatter(x=x,y=elem_f(x,elem_y),mode='lines',name=f'{nom_param}={elem_y}'))
    layout = go.Layout(title=titre,xaxis=dict(title=xlabel),yaxis=dict(title=ylabel))
    fig = go.Figure(data=data, layout=layout)
    return iplot(fig, filename='courbe_param')