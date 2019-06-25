# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:09:26 2019

@author: jazzn
"""

import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly.graph_objs as go
from chempy import balance_stoichiometry
from prettytable import PrettyTable
from colorama import Fore,Style
from ipywidgets import FloatProgress
from IPython.display import display


class Reaction():
    def __init__(self,data,info='pas de description'):
        self.dataframe = pd.DataFrame.from_dict(data)
        self.dataframe.info = info
        self.info = info
        self.bar = {}

    def set_bar(self,esp):
        self.bar[esp] = FloatProgress(value=0,min=0,max=100.0,step=1,description=esp,bar_style='info',orientation='horizontal')
        display(self.bar[esp])
        
    def display_bar(self,esp,x,opt='percent'):
        nmax = max(self.get_qte_final(esp),self.get_qte_ini(esp))
        x=min(max(0,x),self.get_xmax())
        n = self.get_qte(esp,x)
        p = int(100*n/nmax)
        self.bar[esp].description = f"{esp}  ({p}%)" if opt=='percent' else f"{esp}  ({n} mol)"
        self.bar[esp].value = p

    def display_balance_eq(self,opt=1):
        """
        opt = 1 on print sans return
        """
        react = self.dataframe
        eq_bilanR = " + ".join([f'{elem[0]} {elem[1]}' for elem in zip(list(react.loc[react['type']=='r','coef']),list(react.loc[react['type']=='r','esp']))])
        eq_bilanP=" + ".join([f'{elem[0]} {elem[1]}' for elem in zip(list(react.loc[react['type']=='p','coef']),list(react.loc[react['type']=='p','esp']))])
        txt = f'{self.info}\n{eq_bilanR} --> {eq_bilanP}'
        if opt == 1: 
            print(txt)
        else:
            return txt
        
    def display_bar(self,esp,x,endchar=''):
        nmax = self.get_qte_final(esp) if self.get_type(esp) == 'p' else self.get_qte_ini(esp)
        n = self.get_qte(esp,x)
        p=int(100*n/nmax)
        deb,fin='▓'*p,'░'*(100-p)
        print('|'+deb+fin+'| '+esp+' '+str(p)+"%",end='\r')
        if endchar != "":
            print(endchar)
        
    def histogram(self,x):
        xmax = self.get_xmax()
        x = max(min(x,xmax),xmax)
        for esp in self.get_esp_list():
            self.display_bar(esp,x,endchar='\n')
        print('',flush=True)
        
        
    
    def get_balance_eq(self):
        reac0 = set(self.get_reactant_list())
        prod0 = set(self.get_product_list())
        reac,prod = balance_stoichiometry(reac0,prod0)
        reaction = self
        for esp in reac0:
            reaction.set_coef(esp,reac[esp])
        for esp in prod0:
            reaction.set_coef(esp,prod[esp])
        reaction.display_balance_eq()
    
    def get_coef(self,esp):
        return float(self.dataframe.loc[self.dataframe['esp'] == esp,'coef'])
    
    def get_qte_ini(self,esp):
        return float(self.dataframe.loc[self.dataframe['esp'] == esp,'qte_ini'])
    
    def get_reactant_list(self):
        return list(self.dataframe.loc[self.dataframe['type'] == 'r','esp'])
    
    def get_esp_list(self):
        return list(self.get_reactant_list())+list(self.get_product_list())
    
    def get_product_list(self):
        return list(self.dataframe.loc[self.dataframe['type'] == 'p','esp'])
    
    def get_type(self,esp):
        return list(self.dataframe.loc[self.dataframe['esp'] == esp,'type'])[0]
    
    def __repr__(self):
        return self.dataframe.to_string()
    
    def set_qte_ini(self,esp,qte_ini):
        qte_ini = np.abs(qte_ini)
        self.dataframe.loc[self.dataframe['esp'] == esp,'qte_ini']=qte_ini
    
    def set_coef(self,esp,coef):
        self.dataframe.loc[self.dataframe['esp'] == esp,'coef']=coef
    
    def __str__(self):
        return self.__repr__()
    
    def get_xmax(self):
        return np.array([self.get_qte_ini(esp)/self.get_coef(esp) for esp in self.get_reactant_list()]).min()
    
    def get_qte(self,esp,x):
        s = 1. if (self.get_type(esp) == 'p') else -1.
        return self.get_qte_ini(esp) + s*self.get_coef(esp)*x 
    
    def get_qte_final(self,esp):
        return self.get_qte(esp,self.get_xmax())
    

    def plot(self,*esp):
        if isinstance(esp[0],list):
            esp=list(esp[0])
        esp = list(esp)
        xmax = self.get_xmax()
        x = np.linspace(0,xmax)
        rl = " - ".join(list(self.dataframe.loc[(self.dataframe['type']=='r') & (self.dataframe['qte_ini']/self.dataframe['coef'] == xmax),'esp']))
        xlabel,ylabel,titre='avancement x(mol)','quantité (mol)',f'{self.display_balance_eq(opt=0)}'
        courbes = []
        for e in esp:
            if e in self.get_esp_list():
                y = self.get_qte(e,x)
                if e in self.get_reactant_list():
                    if e in rl:
                        label = f'(RL) xmax={xmax} mol'
                    else:
                        label = '(R)'
                else:
                    label = '(P)'
                courbes.append(go.Scatter(x=x,y=y,mode='lines',name=e+label))
        layout= go.Layout(title= titre,xaxis= dict(title= xlabel),yaxis=dict(title= ylabel))
        fig = go.Figure(data=courbes, layout=layout)
        return iplot(fig,filename=titre)

    def display_table(self,x=None,opt='limited'):
        t = PrettyTable()
        if isinstance(x,(int,float)):
            x = float(max(min(self.get_xmax(),x),0))
        colored_esp = []
        for esp in self.get_esp_list():
            colored_esp.append(f"{Fore.GREEN}{esp}{Style.RESET_ALL}" if self.get_type(esp) == 'r' else f"{Fore.RED}{esp}{Style.RESET_ALL}")
        t.field_names = ['équation']+[f"{int(coef)} {esp}" for coef,esp in zip([self.get_coef(esp) for esp in self.get_esp_list()],colored_esp)]
        EI = ['EI x=0']+[f"{qte_ini}" for qte_ini in [self.get_qte_ini(esp) for esp in self.get_esp_list()]]
        t.add_row(EI)
        if opt == 'all':
            if x == None:
                Ei = [f'Ei x']+[f"{qte_ini}{'-' if typ == 'r' else '+'}{'' if coef==1 else int(coef)}x" for qte_ini,coef,typ in zip([self.get_qte_ini(esp) for esp in self.get_esp_list()],[self.get_coef(esp) for esp in self.get_esp_list()],[self.get_type(esp) for esp in self.get_esp_list()])]
            else:
                Ei = [f'Ei x={x}']+[f"{qte}" for qte in [self.get_qte(esp,x) for esp in self.get_esp_list()]]
            t.add_row(Ei)
            xmax = self.get_xmax()
            EF = [f'EF xmax={xmax}']+[f"{qte}" for qte in [self.get_qte_final(esp) for esp in self.get_esp_list()]]
            t.add_row(EF)
        print(t)
        
def check(func,check_value,*arg):
    resultat = func(*arg)
    res =f"Test de votre fonction {func.__name__}("
    for elem in arg:
        if isinstance(elem,float) or isinstance(elem,str) or isinstance(elem,int):
            res+=str(elem)+","
        if isinstance(elem,Reaction):
            res+="reaction,"
    res=res[:-1]
    res+=f")={resultat} , alors que le résultat attendu est {func.__name__}={check_value} "
    if resultat == check_value:
        res+="==> Test réussi"
    else:
        res+="==> Echec ... revoir votre fonction"
    print(res)


    
                    
            

