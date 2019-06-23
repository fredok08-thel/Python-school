# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 23:01:44 2019

@author: jazzn
"""
__autor__ = 'FJ'
from vector import Vector
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from pyfred import f_borne
from random import random

class Parabolic():
    """
    Creation d'une instance de classe Parabolic
    Arguments du constructeur :
        > v0,A0,tmax=1,mass=1,nb=50,g=9.81,coef_frott=0
            v0 (Vector ou tuple)
            A0 (Vector ou tuple)
    Action : Résolution de l'équation du mouvement dans le plan Oxy
        x'' + coef.x' = 0
        y'' + coef.y' = -g
    Méthode:
        .get_TXT(option)
        Si option == 'np' : Retourne les tableaux t(temps), x,y (postions)
        Si option == 'df' : Retourne une dataframe de colonnes t,x,y
        
        .get_TV(option)
        Si option == 'np' : Retourne les tableaux t(temps), vx,vy (vitesses)
        Si option == 'df' : Retourne une dataframe de colonnes t,vx,vy
        
        .get_all(option)
        Si option == 'np' : Retourne les tableaux t(temps), x,y (positions), vx,vy (vitesses)
        Si option == 'df' : Retourne une dataframe de colonnes t,x,y (positions), vx,vy (vitesses)
    """
    def __init__(self,A0,v0,tmax=1,mass=1,nb=50,g=9.81,coef_frott=0):
        if type(v0) == type(Vector()):
            self.v0 = v0
        elif isinstance(v0,tuple):
            self.v0 = Vector(v0)
        if type(A0) == type(Vector()):
            self.A0 = A0
        elif isinstance(A0,tuple):
            self.A0 = Vector(A0)
        self.g = g
        self.tmax = tmax
        self.mass = mass
        self.nb = nb
        self.frott=coef_frott
    
    def solve_ode(self,option='TXY'):
        def eqX(tx,t,coef):
            x,vx = tx
            dtxdt = [vx,-coef*vx]
            return dtxdt
        
        def eqY(ty,t,coef,g):
            y,vy = ty
            dtydt = [vy,-g-coef*vy]
            return dtydt
        t = np.linspace(0,self.tmax,self.nb)
        cond_iniX = [self.A0[0],self.v0[0]]
        cond_iniY = [self.A0[1],self.v0[1]]
        soluceX = odeint(eqX,cond_iniX,t,args=(self.frott,))
        soluceY = odeint(eqY,cond_iniY,t,args=(self.frott,self.g))
        if option.upper() == 'TXY':
            return t,soluceX[:,0],soluceY[:,0]
        elif option.upper() == 'TV':
            return t,soluceX[:,1],soluceY[:,1]
        elif option.upper() == 'ALL':
            return t,soluceX[:,0],soluceY[:,0],soluceX[:,1],soluceY[:,1]
        
    def get_TXY(self,option='np'):
        t,x,y = self.solve_ode('TXY')
        if option == 'np':
            return t,x,y
        elif option == 'df':
            data = pd.DataFrame({'t':t,'x':x,'y':y})
            data.info = 'Mvt parabolique'
            return data
        
    def get_TV(self,option='np'):
        t,vx,vy = self.solve_ode('TV')
        if option == 'np':
            return t,vx,vy
        elif option == 'df':
            data = pd.DataFrame({'t':t,'vx':vx,'vy':vy})
            data.info = 'Mvt parabolique'
            return data    
        
    def get_all(self,option='np'):
        t,x,y,vx,vy = self.solve_ode('all')
        if option == 'np':
            return t,x,y,vx,vy
        elif option == 'df':
            data = pd.DataFrame({'t':t,'x':x,'y':y,'vx':vx,'vy':vy})
            data.info = 'Mvt parabolique'
            return data
        
class Planet_characters:
    """
    """
    def __init__(self,planet_name='Terre', a=149.6e9, e=0.01671022, center_mass=1.989e30):
        """
        ** Initialisateur
        Arguments et valeurs par défauts:
            * planet_name (str): nom de la planete - défaut 'Terre'
            * a (float): demi-grand axe en m - défaut 149.6e9 (demi grand axe de la Terre)
            * e (float): exentricité (grandeur sans unité) - défaut 0.01671022
            * center_mass (float): masse de l'astre central en kg - défaut 1.989e30 (soleil)
        Exemple d'utilisation:
            >mars = Planet_characters(planet_name='Mars',a=....,e=....)
        """
        self.planet_name = planet_name
        self.a = a
        self.e = e
        self.center_mass = center_mass
    
    def get_semi_minor(self):
        return self.a*np.sqrt(1-self.e**2)
    
    def get_period(self):
        return np.sqrt((4*np.pi**2*self.a**3)/(6.67408e-11*self.center_mass))
    
    def get_kinematic(self,deltaT=1):
        """
        retourne une dataframe de la cinématique complete de la planète (self)
        contenant les colonnes t,x,y ou t varie de 0 à T (période) par pas de deltaT
        deltaT est en jours (j)
        
        Exemple : 
            > mars = Planet_kinemat('Mars',a=227939100,e=0.093315) #instanciation de mars
            >mvt_mars = mars.get_kinematic(2) # creer la cinématique de mars sur une période par pas de 2j
        """
        a = self.a
        e = self.e
        deltaTs = deltaT * 86400
        period = self.get_period()
        if (period/deltaTs)>200:
            print(f"votre deltaT={deltaT}j est trop petit par rapport à la période de la planete {self.planet_name} qui est de {period/86400}j")
            print(f"la valeur de deltaT maximale est {period/(200*86400)}j")
            return None
        if (period/deltaTs)<3:
            print(f"votre deltaT={deltaT}j est trop grand par rapport à la période de la planete {self.planet_name} qui est de {period/86400}j")
            print(f"la valeur de deltaT minimale est {period/(10*86400)}j")
            return None
        b = self.get_semi_minor()
        p = a*(1-self.e**2)
        cst = 2*np.pi*a*b/(period*p**2)
        # vitesse aerolaire
        def foo(y,t):
          return cst*(1+e*np.cos(y))**2
        t = np.arange(0,period,deltaTs)
        theta = odeint(foo,0,t)
        columns = ['t','x', 'y']
        planet_result = pd.DataFrame(columns = columns)
        planet_result['t'] = t
        r = p/(1+e*np.cos(theta))
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        planet_result['x'] = x
        planet_result['y'] = y
        planet_result['angle'] = theta
        planet_result.info = self.planet_name
        return planet_result
    
def pendulum(info="pas d'info",length=1,gravity=9.81,vit0=0,theta0=0.2,coef_frot=0,t_max=10,sample=200,repere='low'):
    """ fonction qui retourne une dataframe (cad un tableau contenant t,x,y,theta) de la cinématique d'un pendule
        Liste des arguments:
            info (str): description du pendule - défaut = "pas d'info"
            length (float): longueur du pendule en mètre - défaut = 1
            gravity (float): valeur du champ de pesanteur (valeur de g) - défaut = 9.81
            vit0 (float): vitesse initiale du pendule - défaut = 0
            theta0 (float): angle initial du pendule en radian - défaut = .2
            coef_frot (float): valeur du coefficient de frottement massique - défaut = 0
            t_max (float): durée de la cinématique en secondes - défaut 10
            sample (int): nombre d'enregistrements - défaut = 200
            repere (str): si "low" l'origine du repère est pris à la position la plus basse du pendule
                          si "origin" l'origin du repère correspond au point d'attache du pendule
                          
        Retourne une dataframe contenant les colonnes suivantes :
            	t	theta	x	y

        exemple pour creer la cinématique d'un pendule de 25cm sur la lune
        pendul_lune = pendulum(info='sur la lune',length=0.25,gravity=1.66,sample=400)
    """
    theta0 = theta0 % (2*np.pi)
    t_max = f_borne(t_max,1,100)
    length = f_borne(length,0.01,10)
    vit0 = f_borne(vit0,0,10)
    thetadot0 = vit0/length
    coef_frot = f_borne(coef_frot,0,10)
    sample = f_borne(sample,100,1000)
    gravity = f_borne(gravity,0.1,100)
    w = np.sqrt(gravity/length) 
    txt = f"paramètres de votre pendule après vérifications :\ninfo={info}\nL={length} m \ttheta0={theta0} rad\tvit0={vit0}\ng={gravity} m/s²\tfrot={coef_frot} SI\nt_max={t_max} s\tnb_mesures={sample}\n"
    print(txt)
    def forced_pendulum_equations(y, t, q, w):
        theta, theta_dot = y
        return [theta_dot, -w**2*np.sin(theta) - q * theta_dot]
    t = np.linspace(0,t_max,sample)
    sol = odeint(forced_pendulum_equations, (theta0, thetadot0), t,args=(coef_frot,w))
    theta=sol[:,0]
    if repere.upper()=='LOW':
        x = length*np.sin(theta)
        y = length*(1-np.cos(theta))
    elif repere.upper()=='ORIGIN':
        x = length*np.sin(theta)
        y = -length*np.cos(theta)
    data = pd.DataFrame({'t':t,'theta':theta,'x':x,'y':y})
    data.info=info
    print(f"Voici les premières lignes de votre dataframe\n{data.head()}\n")
    return data

class Radioactivity:
    def __init__(self,info='noyau inconu',N0=100,lamb=0.25,dt=1):
        lamb = max(min(0.9999999, lamb), 1e-20)
        N0 = int(max(min(1e6, N0), 10))
        self.dt = dt
        self.N0 = N0
        self.lamb = lamb
        self.N = N0
        self.t = 0
        self.data = pd.DataFrame({'t':self.t*np.ones(1),'N':self.N*np.ones(1)}) 
        self.data.info = info
        print(f"{self.data.info}\nCaractéristiques:\n*Nombre initiale de noyaux={self.N}\n*lambda={self.lamb}\n*Pas de temps={self.dt}")
    
    def desintegration(self):
        if self.N<=0:
            print("il n'y a plus de noyau")
        else:
            self.t += self.dt
            nb_des = 0
            x = float(self.lamb*self.dt)
            for i in range(int(self.N)):
                if random()<x:
                    nb_des += 1
            self.N -= nb_des
            self.data = self.data.append(pd.DataFrame({'t':self.t*np.ones(1),'N':self.N*np.ones(1)}),ignore_index = True)
            nb_etoile = int(80*self.N/self.N0)
            #print(f"t={self.t}\tnombre de désintégrations={nb_des}\t\tnoyaux restants={self.N}")
            print(f"t={self.t}\t{nb_etoile*'*'}")