# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:03:52 2019

@author: jazzn
"""
from prettytable import PrettyTable
from pyfred import f_input
from IPython.display import clear_output


class Menu():
    def __init__(self,titre="Choix",list_menu=None):
        self.titre = titre
        if isinstance(list_menu,list):
            self.list_menu = list_menu
        else:
            self.list_menu = []
    
    def add_menu(self,sub_titre,func=None):
        if isinstance(sub_titre,str) and ((func == None) or callable(func)):
            self.list_menu.append([sub_titre,func])
        else:
            print("Le titre et/ou la fonction ne convien(ent) pas")
    
    def del_menu(self,s_text):
        """
        supprime le sous menu dont le sous-titre contenant un extrait de s_text
        Exemple: toto_menu est un menu 
        +--------------------+
        | Choix              |
        +--------------------+
        | 0 - Calcul de i2   |
        | 1 - Calcul de n2   |
        | 2 - calcul de toto |
        | 3 - Quitter        |
        +--------------------+
        Pour éliminer le sous menu 'calcul de toto' :
            > toto_menu.del_menu('ul de to') ou bien
            > toto_menu.del_menu('de toto') etc...
        """
        list_titre = [self.list_menu[i][0] for i,_ in enumerate(self.list_menu)]
        for i,elem in enumerate(list_titre):
            if s_text in elem:
                self.list_menu.pop(i)
                break
    
    def show(self):
        txt = PrettyTable()
        txt.field_names=[self.titre]
        txt.align[self.titre] = 'l'
        for i,val in enumerate(self.list_menu):
            v = f"{i} - {val[0]}"
            txt.add_row([v])
        n_max = len(self.list_menu)
        v = f"{n_max} - Quitter"
        txt.add_row([v])
        print(txt) 
        while True:
            choix = f_input("Votre choix","int",(0,n_max))
            if choix == n_max:
                break
            else:
                clear_output()
                print(txt)
                try:
                    self.list_menu[choix][1]()
                except:
                    pass
