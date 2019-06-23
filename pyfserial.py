# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:04:46 2019

@author: jazzn
"""
import pandas as pd
from serial import *
import time

def display_progress(x, tot):
    p=int(100*x/tot)
    deb,fin='▓'*p,'░'*(100-p)
    print('|'+deb+fin+'|'+str(p)+"%",end='\r') 

def read_serial(port="auto",baudrate=9600,timeout = 50,nb_line = 250,sep=";",convert_to='df',convert_as='float',columns_name=None,auto_save=True,filename=None,filetype='xlsx',decimal=','):
    """
    Fonction qui retourne des enregistrement depuis une lecture du port série (ou COM)
    Arguments & défauts:
        * port (str) : nom du port - défaut "auto"
            ** si port = "auto" la fonction recherche le port com automatiquement
        * baudrate (int) : baudrate du port com - défaut 9600 baud
        * timeout (int) en (s) : temps au dessus duquel la tentative de lecture du port com est abandonnée - défaut 50
        * nb_line (int) : nombre de ligne d'enregistrement - défaut 250
        * sep (str) : caractère séparateur des données lues sur le port com - défaut ','
        * convert_to (str) :
            ** si 'df', la fonction retourne une dataframe avec autant de colonnes que de grandeurs lus.
            ** si 'list', la fonction retourne une liste des données lues.
        * convert_as (str) : (valable si convert_to = 'df')
            ** si 'float', les colonnes sont converties en grandeurs numériques
        * columns_name (list ou tuple de str): nom des grandeurs - Défaut None
            ** si columns_name n'est pas utilisée, la fonction recherche automatiquement le nom des colonnes
        * autosave (boolean) : Autorise la sauvegarde vers un fichier - défaut True
        * filename (str) : Nom du fichier sauvegardé - Défaut None
        * filetype (str) : Type du fichier sauvegardé
                            ** si "xlsx" le fichier sera un fichier Excel (défaut)
                            ** si "csv" ........................... csv
        Exemple:
            Le port série envoit des données dont les entêtes sont 't_s','U_V','T_°' 
            ici un temps, une tension et une température séparés par des virgules
            Pour enregister 200 lignes de données.
            >enregistrement = read_serial (port='COM3',nb_line=200,columns_name=('t','U','T'))
            enregistrement est donc une dataframe :
            	t	U	T
            1	0	406	7
            2	10	404	49
            3	20	404	73
            4	31	403	58
            etc...
    """
    if timeout<20:
        timeout = 20
    nb_line = max(min(2000, nb_line), 5)
    if port.lower() == "auto":
        port = serial_ports()
        if port != None:
            port = port[0]
    list_ligne = []
    ligne=''
    st = int(nb_line/25)
    with Serial(port=port, baudrate=baudrate, timeout=timeout, writeTimeout=1) as port_serie:
        if port_serie.isOpen():
            i = 0
            while True:
                if i%st == 0:
                    display_progress(i,nb_line)
                if i>nb_line: 
                    break
                try:
                    ligne = port_serie.readline().decode('ascii').strip()
                    list_ligne.append(ligne.split(sep))
                except UnicodeDecodeError:
                    ligne=''
                i += 1
    port_serie.close()
    print(f"upload from {port} ... finish")
    if convert_to == 'df':
        d = pd.DataFrame(list_ligne)
        d.info = f"upload from :{port}"
        if isinstance(columns_name,(list,tuple)):
            col = columns_name
        else:
            col = d.iloc[0]
        d.drop(0,inplace = True)
        d.columns = col
        if convert_as == 'float':
            _c = d.columns[d.dtypes.eq(object)]
            d[_c] = d[_c].apply(pd.to_numeric, errors='coerce')
        if auto_save:
            if filetype.lower() == 'xlsx':
                file = "file_"+time.strftime("%d-%m-%Y@%H-%M")+".xlsx" if filename == None else filename
                if ".xlsx" not in file:
                    file = file +".xlsx"
                try:
                    d.to_excel(index=False,excel_writer=file)
                    print("Saved file ..."+file)
                except:
                    print("unable to save file")
            else:
                file = "file_"+time.strftime("%d-%m-%Y@%H-%M")+".csv" if filename == None else filename
                if ".csv" not in file:
                    file = file +".csv"
                try:
                    d.to_csv(index=False,path_or_buf=file,sep=';',decimal='.')
                    print("Saved file ..."+file)
                except:
                    print("unable to save file")
        return d
    elif convert_to == 'list':
        return list_ligne
           
            
def serial_ports():
    """ Liste les ports com uniquement pour windows
    """
    ports = [f'COM{i+1}'for i in range(256)]
    resultats = []
    for port in ports:
        try:
            p = Serial(port)
            p.close()
            resultats.append(port)
        except (OSError, SerialException):
            pass
    if resultats:
        print("List COM ports used")
        for e in resultats:
            print(e)
        return resultats
    else:
        print("no COM port detected")
        return None