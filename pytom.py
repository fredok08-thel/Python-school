# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:43:32 2019

@author: jazzn
"""

from mendeleev import element
from colorama import Fore,Style
import numpy as np
import plotly.graph_objs as go
from plotly.offline import  iplot


def histogram(Z,A,opt='mass'):
    if opt=='mass':
        data = [go.Bar(
                x=[f'atome {get_element_name(Z)}', 'Protons', 'Neutrons','Electrons'],
                y=[A*1.673e-27, Z*1.673e-27, (A-Z)*1.673e-27,Z*9.109e-31]
        )]
        layout = go.Layout(title='Répartition en masse')
    else:
        data = [go.Bar(
                x=[f'atome {get_element_name(Z)}', 'Noyau'],
                y=[get_atomic_radius(Z), 1.2e-15*A**(1/3)]
        )]
        layout = go.Layout(title="Rayon de l'atome et de son noyau")
    return iplot(go.Figure(data=data, layout=layout), filename='basic-bar')

def display_compo(Z,A):
    assert A>=Z, "A doit être supérieur ou égal à Z"
    assert A>0 and Z>0, "A et Z doivent être positif non nul"
    A,Z = int(A),int(Z)
    c = int(np.ceil(np.sqrt(A+Z)))
    x,y=[],[]
    for num in range(Z):
        x.append(num%c)
        y.append(num//c)
    trace_p = go.Scatter(x =x,y = y,mode = 'markers',marker = dict(size = 60,symbol=3),name='Protons')
    x,y=[],[]
    for num in range(Z+1,A+1):
        x.append(num%c)
        y.append(num//c)
    trace_n = go.Scatter(x =x,y = y,mode = 'markers',marker = dict(size = 60),name='Neutrons')
    x,y=[],[]
    for num in range(A+1,A+Z+1):
        x.append(num%c)
        y.append(num//c)
    trace_e = go.Scatter(x =x,y = y,mode = 'markers',marker = dict(size = 30,symbol=32),name='Electrons')
    layout = dict(title = 'composition de :'+get_element_name(Z),
              yaxis = dict(showgrid=False,showline=False,zeroline=False),
              xaxis = dict(showgrid=False,showline=False,zeroline=False)
             )
    fig = dict(data=[trace_p,trace_n,trace_e], layout=layout)
    return iplot(fig,filename='basic-scatter')

def pie_chart(values,labels=[]):
    trace = go.Pie(labels=labels,values=values)
    return iplot([trace],filename='pie chart')

def get_element_name(Z):
    """
    argument(s):
        > Z (int): atomic number
    return:
        > name of element (str)
    Exemple:
        > get_name(6)
        > 'carbon'
    """
    return element(Z).name

def get_element_symbol(Z):
    """
    argument(s):
        > Z (int): atomic number
    return:
        > symbol of element (str)
    Exemple:
        > get_name(6)
        > 'C'
    """
    return element(Z).symbol

def get_atomic_radius(Z):
    """
    argument(s):
        > Z (int): atomic number
    return:
        > atomic radius in m
    """
    return element(Z).atomic_radius*1e-12

def check_func(func,check_value,*arg,**d):
    """
    Arguments:
        func (function): fonction a tester
        check_value : valeur correcte
        compare_mode (str): 
            * "equality" utilise ==
            * "is_close" utilise np.is_close
    """
    resultat = func(*arg)
    compare_mode = "isclose" if isinstance(resultat,(float,complex)) else ""
    res =f"Test de votre fonction {func.__name__}("
    for elem in arg:
        if isinstance(elem,(float,str,int)):
            res+=str(elem)+","
        try:
            if isinstance(elem,Reaction):
                res+="reaction,"
        except:
            pass
    res=res[:-1]
    res+=f")={resultat} , alors que le résultat attendu est {func.__name__}={check_value} "
    if compare_mode == 'isclose':
        if np.isclose(resultat,check_value,atol=d.get('atol',0),rtol=d.get('rtol',1e-3)).all():
            res+=f"==> {Fore.GREEN}Test réussi{Style.RESET_ALL}"
        else:
            res+=f"==> {Fore.RED}Echec ... revoir votre fonction{Style.RESET_ALL}"  
    else:
        if resultat == check_value:
            res+=f"==> {Fore.GREEN}Test réussi{Style.RESET_ALL}"
        else:
            res+=f"==> {Fore.RED}Echec ... revoir votre fonction{Style.RESET_ALL}"
    print(res)