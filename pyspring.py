# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:48:12 2019

@author: jazzn
"""
from vpython import *
from numpy import pi
from prettytable import PrettyTable
from traceback import extract_stack

class Ressort:
    @classmethod
    def __non_null(cls,x,default=1,option='positiv'):
        if x == 0:
            return default
        if option == 'positiv':
            return abs(x)
        else:
            return x
    @classmethod
    def __normalize(cls,l):
        if isinstance(l,list) or isinstance(l,tuple):
            if len(l)==3:
                norm = [float(i)/sum(l) for i in l]
                return vector(norm[0],norm[1],norm[2])

    
    def __init__(self,masse=1,pos_x=0,k=10,nb_spires=10,rho=5,planet = 'terre',g = 9.81,couleur=color.red,info=""):
        (_,_,_,text)=extract_stack()[-2]
        def_name = text[:text.find('=')].strip()
        try:
            self.__del__()
        except:
            pass
        self.pos_x = int(pos_x)*3
        self.masse = Ressort.__non_null(masse)
        self.nb_spires = Ressort.__non_null(nb_spires,10)
        self.rho = Ressort.__non_null(rho)
        self.planet = planet
        if isinstance(couleur,list) or isinstance(couleur,tuple):
            couleur = Ressort.__normalize(couleur) 
        self.couleur=couleur
        self.k = Ressort.__non_null(k)
        self.g = Ressort.__non_null(g)
        self.info = def_name if info == "" else info
        self.long = self.masse*self.g/self.k
        self.rayon = (3*self.masse/(4*pi*self.rho))**(1/3)
        self.cylinder = cylinder(canvas=scene,pos=vector(self.pos_x,0,0),axis=vector(0,0.2,0),radius=1.5,color = color.white)
        self.spring = helix(canvas=scene,pos=vector(self.pos_x,0,0), axis=vector(0,-self.long,0), radius=0.5,coils=nb_spires)
        self.mass = sphere(canvas=scene,pos=self.spring.pos+self.spring.axis,radius=self.rayon,color=couleur)

    def get_lenght(self):
        return self.long
    
    def set_mass(self,masse):
        self.masse = Ressort.__non_null(masse,1)
        self.update()
    
    def set_pos_x(self,pos_x):
        self.pos_x = pos_x
        self.update()
    
    def set_k(self,k):
        self.k = Ressort.__non_null(k)
        self.update()
    
    def set_g(self,g):
        self.g = Ressort.__non_null(g)
        self.update()
    
    def set_color(self,color):
        self.couleur = color
        self.update()
    
    def set_rho(self,rho):
        self.rho = Ressort.__non_null(rho)
        self.update()
    
    def display(self):
        return scene.updates
    
    def is_same_size(self,other):
        if abs(self.long - other.long)<=0.025:
            print(f'les 2 ressorts ont même longueur={self.long} et {other.long}')
        else:
            print(f"les 2 ressorts n'ont pas la même longueur {self.long} et {other.long}")
    
    def __del__(self):
        self.cylinder.visible = False
        self.spring.visible = False
        self.mass.visible = False
        self.update()
        
    def update(self):
        self.long = self.masse*self.g/self.k
        self.rayon = (3*self.masse/(4*pi*self.rho))**(1/3)
        self.spring.pos = vector(self.pos_x,0,0)
        self.spring.axis = vector(0,-self.long,0)
        self.spring.coils = self.nb_spires
        self.mass.pos = self.spring.pos+self.spring.axis
        self.mass.radius=self.rayon
        self.mass.color=self.couleur
        self.cylinder.pos = vector(self.pos_x,0,0)
        return scene.updates
    
    def __repr__(self):
        d = {'info':self.info,'masse (kg)':self.masse,'k (N/m)':self.k,'longueur (m)':self.long,'g (m/s²)':self.g,'rho (kg/m3)':self.rho,'position':self.pos_x/3,'Planet':self.planet}
        txt = PrettyTable()
        txt.field_names = ["attribut","valeur"]
        txt.align["attribut"] = 'l'
        for cle,val in d.items():
            txt.add_row([cle,val])
        return str(txt)
        
       
        