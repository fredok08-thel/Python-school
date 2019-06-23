__autor__ = 'FJ'
#A executer en premier
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly import figure_factory as ff
from plotly.offline import  iplot
from plotly import tools
from scipy import stats
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.optimize import curve_fit
import IPython as ip
from uncertainties import ufloat
from time import sleep
from colorama import Fore, Back, Style
from inspect import signature,getsource
from astroquery.jplhorizons import Horizons

SPY = []
SPY_KEY = 9 #clé pour l'utilisation de check_graph ou check_op ou ...
__version__ = '1.0.4'

def enable_plotly_in_cell():    
  """
  fonction a mettre pour afficher un graphique plotly uniquement sur google.colab
  """
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  init_notebook_mode(connected=False)

def reset_spy():
    SPY = []


def scatterPolar(self,other_df=None, theta='theta', r='R', titre="",
              unsur=1, style_tracer='o',
              color='auto', quadril_r=0, quadril_theta=0):
    couleur=['red', 'blue', 'green', 'grey', 'darkgrey', 'gold', 'black',
             'pink', 'darkturquoise', 'lightblue', 'purple', 'maroon', 'violet', 'mistyrose']
    datf = [self]
    r = format_to_valid_y(r)
    if not isinstance(other_df, list):
        other_df = [other_df]
    for elem in other_df:
        if isinstance(elem, pd.DataFrame):
            datf.append(elem)
    courbes = []
    i = 0
    symbol, s = ["circle",17,3,4,29,29,23,24,10,11,15], list(r'o*+x[]<>/\%')
    for data in datf:
        if not check_info(data):
            print(f"Avertissement - dataframe sans nom\nEcrire ma_dataframe.info='quelque chose'")
            data.info = "sans info"
        nom = data.info
        for ordonnee in r:
            a_virer = False
            if isinstance(ordonnee,(float, int)):
                a_virer = True
                txt=f'_val={ordonnee:.3e}'
                data[txt] = pd.Series(np.ones(len(data)) * float(ordonnee))
                ordonnee = txt
                r[i] = txt
            xyz=go.Scatterpolar(theta=data[theta][::unsur],r=data[ordonnee][::unsur],
                               name=f"{ordonnee}/({nom})")
            if style_tracer in s:
                xyz['mode']="markers"
                xyz['marker']['symbol'] = symbol[s.index(style_tracer)]
            elif style_tracer in ['-o','o-']:
                xyz['mode']="lines+markers"
                xyz['line']['shape']='spline'
            else:
                xyz['mode']="lines"
                xyz['line']['shape']='spline'
            if color=='auto':
                xyz['marker']['color']=couleur[i]
            elif color=='rainbow':
                xyz['marker']['color']=data[ordonnee][::unsur]
            else:
                xyz['marker']['color']=color
            courbes.append(xyz)
            i = (i + 1)%len(couleur)
            if a_virer:
                data.drop(columns=[txt],inplace=True)
    i = (i + 1)%len(couleur)
    if titre=="":
        try:
            titre="Courbe(s) relative(s) à "+" ; ".join([data.info for data in datf])
            tit_ok = True
        except:
            titre="Pas de titre"
            tit_ok = False
    layout= go.Layout(title= titre)
    fig = go.Figure(data=courbes, layout=layout)
    set_spy(x=r,y=theta)
    return iplot(fig,filename=titre)

def histogram(self,x='x',titre='pas de titre',orientation='vertical',histnorm=""):
   """
   Trace un histogramme de la grandeur 'x'
   Arguments:
       * x (str ou [str]): Nom de la ou des colonnes dont on veux l'histogramme
       * titre (str) : titre de l'histogramme
       * orientation (str) : indique l'orientation de l'histogramme 
           > 'vertical' (défaut) ou 'horizontal' 
       * histnorm (str): Normalise l'histogramme
           > "" (défaut) : pas de normalisation
           > "percent" : normalisation en pourcentage
    Exemple:
        'x' et 'y' sont des noms de colonnes de la dataframe df. Pour tracer
        les histogrammes de 'x' et 'y'
        >df.histogram(x=['x','y'])
   """
   x = format_to_valid_y(x)
   courbes=[]
   for elem in x:
       if orientation.lower()=='vertical':
           courbes.append(go.Histogram(x=self[elem],name=elem,histnorm=histnorm))
       else:
           courbes.append(go.Histogram(y=self[elem],name=elem,histnorm=histnorm))
   layout = go.Layout(barmode='overlay')
   fig = go.Figure(data=courbes, layout=layout)
   return iplot(fig,filename='overlaid histogram')         

           
def scatter2D(self, other_df=None, x='t', y='X', titre="",
              ortho='auto', unsur=1, style_tracer='o',
              color='auto', shape=None, origin_df=None, quadril_x=0, quadril_y=0,
              subplots=False, fill='', xlabel='', ylabel=''):
    """
        Méthode appliquée à une dataframe -> retourne un graphique plotly en 2D
        Arguments, significations et valeurs par defaut        
        * x (str): Nom unique de l'abcisse (ex x='t') - Défaut x='t'
        * y (str ou [str]): Nom(s) de la/des ordonnées (ex y=['X','Y',85.6]) 
                                accepte les valeurs (ex 8.31) - Défaut y='X'
        * titre (str): titre du graphique - Défaut titre=''
        * x_label, y_label (str) : Noms des axes x et y - Défaut ''
        * ortho ('ortho' ou 'auto') : repère normé => ortho='ortho' 
                    sinon ortho='auto' (Défaut ortho='auto')
        * unsur (int): Affiche 1 point sur unsur - Défaut unsur=1
        * style_tracer (str): Défini le style du tracé parmis 'o','-o','--'
                        Défaut style_tracer='o'
        * color (str) : Défini la couleur - Défaut color='auto'
                        Spécial 'rainbow'
        * shape [dict]: Permet d'inserer une forme géométrique sur le graphique
                        Défaut shape=None
               ex pour tracer le soleil de rayon 600000 km=> 
          shape=dict(type='circle',x0=-300000,x1=300000,y0=-300000,y1=300000) 
        * other_df (dataframe): Nom d'une autre dataframe à tracer possédant les mêmes colonnes.
        * subplots (booleen) : Si True, organise vos graphiques en plusieurs sous_graphiques. Défaut = False
        * fill (str) : Permet de colorier en dessous d'une courbe depuis y=0 ('tozeroy') ou entre 2 courbes ('tonexty') - Défaut ''
        * origin_df (dataFrame): définie la dataframe qui sera l'origine du tracé, 
                - conditions : origin_df et la dataframe (self) doivent avoir la même colonne de temps, elles doivent être
                synchrone. Elles doivent avoir des colonnes positions x,y ayant les mêmes noms.
                - Par défaut c'est le repère ou l'on a définie la dataframe (self)
        Exemple 1 : 'mars' et 'phobos' sont des dataframes  possédant 'x' et 'y' comme colonnes. Pour tracer y=f(x) de Mars
                  et de son satellite Phobos
                  >mars.scatter2D(x='x',y='y',other_df=phobos,ortho='ortho',color='rainbow')      
                  
        Exemple 2 : 'ballon' est une dataframe (mouvement d'un ballon) possédant 't','x' et 'y' comme colonnes - Pour tracer
                  les courbes x=f(t), y=f(t):
                  >ballon.scatter2D(x='t',y=['x','y'],titre='équations horaires du mouvement')
        
        Exemple 3 : 'io', 'calisto' et 'jupyter' sont des dataframes synchrones dans le temps et ayant des noms de colonnes identiques - Pour 
                    tracer la trajectoire de io et callisto dans le référentiel jovien:
                    >io.scatter2D(other_df=callisto,origin_df=jupiter,x='x',y='y',titre='trajectoire de io par rapport à Jupyter')
    """
    couleur=['red', 'blue', 'green', 'grey', 'darkgrey', 'gold', 'black',
             'pink', 'darkturquoise', 'lightblue', 'purple', 'maroon', 'violet', 'mistyrose']
    datf = [self]
    y = format_to_valid_y(y)
    if not isinstance(other_df, list):
        other_df = [other_df]
    for elem in other_df:
        if isinstance(elem, pd.DataFrame):
            datf.append(elem)
    origin = False
    if isinstance(origin_df, pd.DataFrame):
        origin = True
    courbes = []
    i = 0
    symbol, s = ["circle",17,3,4,29,29,23,24,10,11,15], list(r'o*+x[]<>/\%')
    for data in datf:
        if not check_info(data):
            print(f"Avertissement - dataframe sans nom\nEcrire ma_dataframe.info='quelque chose'")
            data.info = "sans info"
        nom = data.info
        for ordonnee in y:
            a_virer = False
            if isinstance(ordonnee,(float, int)):
                a_virer = True
                txt=f'_val={ordonnee:.3e}'
                data[txt] = pd.Series(np.ones(len(data)) * float(ordonnee))
                ordonnee = txt
                y[i] = txt
            if not origin:
                xyz=go.Scatter(x=data[x][::unsur],y=data[ordonnee][::unsur],marker=dict(size=4),
                               name=f"{ordonnee}/({nom})"[:50])
            else:
                xyz=go.Scatter(x=data[x][::unsur]-origin_df[x][::unsur],y=data[ordonnee][::unsur]-origin_df[ordonnee][::unsur],
                                   marker=dict(size=4),name=f"{ordonnee}({nom})"[:50])
                xyz_origin = go.Scatter(x=np.zeros(1),y=np.zeros(1),marker=dict(size=4),name=f"{origin_df.info}"[:50])
            if style_tracer in s:
                xyz['mode']="markers"
                xyz['marker']['symbol'] = symbol[s.index(style_tracer)]
            elif style_tracer in ['-o','o-']:
                xyz['mode']="lines+markers"
                xyz['line']['shape']='spline'
            elif style_tracer == ':':
                xyz['mode']="lines"
                xyz['line']['dash'] = 'dot'
            elif style_tracer == ';':
                xyz['mode']="lines"
                xyz['line']['dash'] = 'dash'
            else:
                xyz['mode']="lines"
                xyz['line']['shape'] = 'spline'
            if color=='auto':
                xyz['marker']['color']=couleur[i]
            elif color=='rainbow':
                xyz['marker']['color']=data[ordonnee][::unsur]
            else:
                xyz['marker']['color']=color
            if fill != '':
                xyz['fill'] = fill
            courbes.append(xyz)
            if origin:
               courbes.append(xyz_origin) 
            i = (i + 1)%len(couleur)
            if a_virer:
                data.drop(columns=[txt],inplace=True)
    if titre=="":
        try:
            titre="Courbe(s) relative(s) à "+" ; ".join([data.info for data in datf])
            tit_ok = True
        except:
            titre="Pas de titre"
            tit_ok = False
    x_label = x if xlabel == '' else xlabel
    y_label = " | ".join(y) if ylabel == '' else ylabel
    layout= go.Layout(title= titre,xaxis= dict(title = x_label,nticks=quadril_x),yaxis=dict(nticks=quadril_y,title=y_label ))
    if ortho=='ortho':
        layout['xaxis']['constrain'] ='domain'
        layout['yaxis']['scaleanchor']='x'
    if shape != None:
        if not isinstance(shape,list):
            shape=[shape]
        layout['shapes']=shape
    if subplots:
        nb_rows = len(courbes) // 2 + len(courbes) % 2
        if tit_ok:
            titres = tuple([tit for tit in titre.split('+')][1:])
        fig = tools.make_subplots(rows=nb_rows, cols=2, subplot_titles=titres)
        for idx,curve in enumerate(courbes):
            fig.append_trace(curve, 1+int(idx/2), 1+idx%2)
            fig['layout'][f"xaxis{1+idx}"].update(title = x)
            fig['layout'][f"yaxis{1+idx}"].update(title = f"{curve['name']}")
    else:
        fig = go.Figure(data=courbes, layout=layout)
    set_spy(x=x,y=y,ortho=ortho)
    return iplot(fig,filename=titre)

def set_spy(x, y, z='', ortho='auto'):
    if z == '':
        if ortho == 'ortho':
            SPY.append({'method':'scatter2D', 'x':x,'y':y_to_str(y), 'ortho':'ortho'})
        else:
            SPY.append({'method':'scatter2D', 'x':x, 'y':y_to_str(y)})
    else:
        SPY.append({'x':x, 'y':y_to_str(y),'z':z})

def scatter3D(self, other_df=None, x='X', y='Y', z='Z', titre="un titre", unsur=1,color='auto',origin_df=None,style_tracer='o'):
    """
        Méthode/Fonction appliquée à une dataframe -> retourne un graphique plotly en 3D
        Arguments, signification et valeurs par defauts
        Par convention : on appelle 'self' la dataframe qui appele la méthode. 
        * x,y,z (str): Noms des coordonnées uniques x,y,z qui sont des colonnes de la dataframe self  - Défauts x='X',y='Y',z='Z'
        * titre (str): titre du graphique - Défaut titre='un titre'
        * unsur (int): Affiche 1 point sur unsur - Défaut unsur=1
        * style_tracer (str): Défini le style du tracé parmis 'o','-o','--' - Défaut style_tracer='o'
        * color (str): Défini la couleur - Défaut color='auto' - Spécial 'rainbow'
        * other_df (dataframe): Nom d'une autre dataframe qui possède les mêmes colonnes x,y,z
        * origin_df (dataFrame): définie la dataframe qui sera l'origine du tracé, 
                - conditions : origin_df et la dataframe (self) doivent avois la même colonne de temps, elles doivent être
                synchrones. Elles doivent avoir des colonnes positions x,y,z ayant les mêmes noms.
                - Par défaut l'origine est le repère ou l'on a définie le mvt de la dataframe (self)
        
        Exemple 1  : 'mars' et 'phobos' sont des dataframes possédant 'x','y','z' comme colonnes - Pour tracer la trajectoire de Mars
                  >mars.scatter3D(x='x',y='y',z='z',titre='un super titre')    
        Exemple 2 :  'triton' et 'uranus' sont des dataframes possédant 'x','y','z' comme coordonnées de positions et 
                    synchrone sur 't'. Pour tracer la trajectoire de triton dans le repère d'uranus.
                  >triton.scatter3D(x='x',y='y',z='z',origin_df=uranus)

    """
    
    symbol, s = ['circle', 'circle-open', 'square', 'square-open','diamond', 'diamond-open', 'cross', 'x'],list(r'oO[]<>/x')
    datf = [self]
    if not isinstance(other_df, list):
        other_df = [other_df]
    for elem in other_df:
        if isinstance(elem, pd.DataFrame):
            datf.append(elem)
    trace = []
    couleur = ['red', 'blue','green' , 'grey','darkgrey','gold','black',
             'pink','darkturquoise','lightblue','purple','maroon','violet','mistyrose']
    origin = False
    if isinstance(origin_df,pd.DataFrame):
        origin = True
    for i, df in enumerate(datf):
        if not check_info(df):
            df.info = "pas d'info"
        if not origin:
            xyz=go.Scatter3d(x=df[x][::unsur], y=df[y][::unsur], z=df[z][::unsur], mode='markers', marker=dict(colorscale='Rainbow',
                                                    size=3), name=df.info)
        else:
            xyz=go.Scatter3d(x=df[x][::unsur]-origin_df[x][::unsur], y=df[y][::unsur]-origin_df[y][::unsur],
                             z=df[z][::unsur]-origin_df[z][::unsur], mode='markers', marker=dict(colorscale='Rainbow',
                                                    size=3), name=df.info)
            xyz_origin = go.Scatter3d(x=np.zeros(1), y=np.zeros(1),
                             z=np.zeros(1), mode='markers', marker=dict(colorscale='Rainbow',
                                                    size=5), name=origin_df.info)
        if color == 'auto':
            xyz['marker']['color'] = couleur[i]
        elif color == 'rainbow':
            xyz['marker']['color'] = df[x]
            #xyz['marker']['colorscale']='Rainbow'
        else:
            xyz['marker']['color'] = color
        if style_tracer in s:
            xyz['mode']="markers"
            xyz['marker']['symbol'] = symbol[s.index(style_tracer)]
        elif style_tracer in ['-o','o-']:
            xyz['mode']="lines+markers"
        else:
            xyz['mode']="lines"
        trace.append(xyz)
        if origin:
            trace.append(xyz_origin) 
    layout = go.Layout(title=titre,margin=dict(l=0,r=0,b=0,t=0),showlegend=True,
                       scene = dict(aspectmode='data',xaxis = dict(title=x), 
                                    yaxis = dict(title=y), zaxis=dict(title=z)))
    fig = go.Figure(data=trace, layout=layout)
    SPY.append({'method': 'scatter3D','x':x,'y':y,'z':z})
    return iplot(fig, filename=titre)

def arrondir_cs(x,cs = 1):
    """
    Fonction -> retourne l'arrondi supérieur d'une valeur x avec cs chiffres significatifs (float)
    Arguments:
    * x (float): Valeur à arrondir
    * cs (int): nombre de chiffres significatifs - Défaut cs=1
    """
    cs = min(max(cs, 1), 10)
    parts = ("%e" % x).split('e') 
    exposant = int(parts[1])
    mantisse = np.ceil(float(parts[0])*10**(cs-1))/10**(cs-1)
    return (mantisse*10**exposant)

def get_student_k(n,percent):
    x = 1-(1 - percent)/2 #2tails
    return stats.t.ppf(x,n-1)

def display_U(self,grandeur,cs=1,percent=0.95):
    """
    Fonction ou méthode appliquée à une dataframe -> retourne l'incertitude d'une grandeur U(grandeur) contenue dans une 
    dataframe avec cs chiffres significatifs sous forme d'un texte (str)
    Arguments:
    grandeur (str): nom de la colonne de la dataframe dont on veux l'incertitude (ex: grandeur='R')
    * cs (int): nombre de chiffres significatif - Défaut cs=1
    * perdent (float): niveau de confiance - Défaut 0.95 dsoit 95%
    """

    cs=int(min(max(cs, 1),2))
    moy=self[grandeur].mean()
    std=self[grandeur].std()
    nb=self[grandeur].count()
    k = get_student_k(nb,percent)
    u0=std*k/np.sqrt(nb)
    u=arrondir_cs(u0,cs)
    x=ufloat(moy,u)
    relat=np.abs(arrondir_cs(u/moy,2)*100)
    if cs==1:
        txt=f'grandeur/incertitude \n{grandeur}={x:.1u} à {100*percent}% avec k={k:.4}'
    elif cs==2:
        txt=f'grandeur/incertitude \n{grandeur}={x:.2u} à {100*percent}% avec k={k:.4}'
    txt+='\n'+f'incertitude relative = {relat:.2}%'
    txt+="\nsans arrondi :"
    txt+=f'{grandeur}={moy} et U({grandeur})={u:}'
    txt+='\n\n'
    return txt

def draw_vectors(self,point=('&','&'),vector=('vx','vy'),unsur=1,scalvect=1,titre="sans titre",quadril_x=0,quadril_y=0):
    if (point == ('&','&')) and len(vector)==3:
        point = ('&','&','&')
    if len(point) == len(vector) == 2:
        x,y = point
        ux,uy = vector
        self.vector(x=x,y=y,ux=ux,uy=uy,unsur=unsur,scalvect=scalvect,titre=titre,quadril_x=quadril_x,quadril_y=quadril_y)
    elif len(point) == len(vector) == 3:
        x,y,z = point
        ux,uy,uz = vector
        self.vector3D(x=x,y=y,z=z,ux=ux,uy=uy,uz=uz,unsur=unsur,scalvect=scalvect,titre=titre,quadril_x=quadril_x,quadril_y=quadril_y)
    else:
        print(f'votre point={point} et vector={vector} ne conviennent pas')
                

def vector(self,ux,uy,x='&',y='&',unsur=1,scalvect=1,titre="sans titre",quadril_x=0,quadril_y=0):
    """
        Méthode/Fonction appliquée à une dataframe -> retourne la trajectoire avec des vecteurs.
        Conditions d'utilisation :
        
        Arguments, signification et valeurs par défauts:
        * x,y (str) : Noms des points d'applications des vecteurs - défaut 'x' et 'y'
        * ux,uy (str) : Noms des coordonnées uniques du vecteur à tracer (ex ux='AX', uy='AY')
        * titre (str): titre du graphique - Défaut titre='un titre'
        * unsur (int): Affiche 1 vecteur sur unsur - Défaut unsur=1
        * scalvect (float): echelle de tracé des vecteurs - Défaut scalvect=1.0
        * quadril_x,quadril_y (int) : Densité du quadrillage (défaut 0,0)
    Exemple 1: Utilisation pour visualiser le vecteur vitesse (VX,VY) de la planètes Mars (dataFrameMars)
        >dataFrameMars.vector(ux='VX',uy='VY',unsur=2)
    Exemple 2: Visualiser le vecteur accélération 'ax','ay' de la dataframe ballon
        >ballon.vector(ux='ax',uy='ay')
    """
    xx,yy='',''
    #recherche du x,y par defaut
    if (x=='&' and y=='&'):
        list_x='x','X','x_mod','X_mod' #valeurs préalablement testées
        list_y='y','Y','y_mod','Y_mod'
        for lx in list_x:
            if lx in list(self):
                xx=lx
                break
        for ly in list_y:
            if ly in list(self):
                yy=ly
                break
    if (x in self) and (y in self):
        xx, yy = x, y
    if xx=='' or yy=='':
        print(f"'X ou x' ou 'Y ou y' ne sont pas des colonnes de votre dataFrame, il faut spécifier une origine à vos vecteurs en utilisant x='**' et y='**' ou ** sont parmis\n{self.columns}")
        return None
    scal_espace=np.abs((self[xx].max()+self[yy].max())/(self[ux].max()+self[uy].max()))
    quiver_fig = ff.create_quiver(self[xx][::unsur],self[yy][::unsur], self[ux][::unsur], self[uy][::unsur],
                       scale=scal_espace*scalvect,
                       arrow_scale=0.08, # Sets arrow scale
                       name=f'vecteur ({ux},{uy})',
                       angle=np.pi/12,
                       line=dict(width=1))
    points = go.Scatter(x=self[xx],y=self[yy],mode='markers',marker=dict(size=4,color='grey'),name=f"Point ({xx},{yy})")
    layout= go.Layout(title= titre,xaxis= dict(constrain='domain',title= xx,nticks=quadril_x),yaxis=dict(scaleanchor='x',title= yy,nticks=quadril_y))
    quiver_fig.add_trace(points)
    quiver_fig['layout'].update(layout)
    SPY.append({'method': 'vector','ux':ux,'uy':uy,'x':xx,'y':yy})
    return iplot(quiver_fig, filename='vecteur')

def vector3D(self,ux,uy,uz,x='&',y='&',z='&',unsur=1,scalvect=1,titre="sans titre",quadril_x=0,quadril_y=0):
    """
        Méthode/Fonction appliquée à une dataframe -> retourne la trajectoire avec des vecteurs.
        
        Arguments, signification et valeurs par défauts:
        * x,y,z (str) : Noms des points d'applications des vecteurs
        * ux,uy,uz (str) : Noms des coordonnées uniques du vecteur à tracer (ex ux='AX', uy='AY', uz='AZ')
        * titre (str): titre du graphique - Défaut titre='un titre'
        * unsur (int): Affiche 1 vecteur sur unsur - Défaut unsur=1
        * scalvect (float): echelle de tracé des vecteurs - Défaut scalvect=1.0
    Exemple:  Utilisation pour visualiser le vecteur vitesse (VX,VY,VZ) de la planètes Mars (dataFrameMars)
                dont les points d'application ont pour coordonnées les colonnes 'X','Y','Z' 
        >dataFrameMars.vector3D(x='X',y='Y',z='Z',ux='VX',uy='VY',uz='VZ',unsur=2)
    """
    xx,yy,zz='','',''
    #recherche du x,y par defaut
    if (x=='&' and y=='&' and z=='&'):
        list_x='x','X','x_mod','X_mod' #valeurs préalablement testées
        list_y='y','Y','y_mod','Y_mod'
        list_z='z','Z','z_mod','Z_mod'
        for lx in list_x:
            if lx in list(self):
                xx=lx
                break
        for ly in list_y:
            if ly in list(self):
                yy=ly
                break
        for lz in list_z:
            if lz in list(self):
                zz=lz
                break
    if (x in self) and (y in self) and (z in self):
        xx, yy, zz = x, y, z
    if xx=='' or yy=='' or zz=='':
        print(f"'X ou x' ou 'Y ou y' ou 'Z ou z' ne sont pas des colonnes de votre dataFrame, il faut spécifier un point d'application à vos vecteurs en utilisant x='**' et y='**' ou ** sont parmis\n{self.columns}")
        return None
    scal=scalvect*np.abs((self[xx].max()+self[yy].max()+self[zz].max())/(self[ux].max()+self[uy].max()+self[uz].max()))
    xyz=go.Scatter3d(x=self[xx], y=self[yy], z=self[zz], mode='markers', marker=dict(colorscale='Rainbow',
                                                    size=3), name=self.info)
    trace = []
    trace.append(xyz)
    for i,coordonnees in enumerate(zip(self[xx][::unsur],self[yy][::unsur],self[zz][::unsur],self[ux][::unsur],self[uy][::unsur],self[uz][::unsur])):
        vec_x=[coordonnees[0],coordonnees[0]+scal*coordonnees[3]]
        vec_y=[coordonnees[1],coordonnees[1]+scal*coordonnees[4]]
        vec_z=[coordonnees[2],coordonnees[2]+scal*coordonnees[5]]
        vector=go.Scatter3d(x=vec_x,y=vec_y,z=vec_z,marker = dict( size = 1),line = dict(width = 2),name='')
        trace.append(vector)
    layout = go.Layout(title=titre,margin=dict(l=0,r=0,b=0,t=0),scene = {'aspectmode':'data'},showlegend=False)
    fig = go.Figure(data=trace, layout=layout)
    SPY.append({'method': 'vector3D','ux':ux,'uy':uy,'uz':uz,'x':xx,'y':yy,'z':zz})
    return iplot(fig, filename='vecteur3D')

def get_kinematic(self,xyzt,**d):
    xyzt = format_to_valid_y(xyzt)
    dt = xyzt[-1]
    option = d.get('opt',(1,0))
    #create velocity
    txt_v,txt_a, txt_r = '','',''
    for elem in xyzt[:len(xyzt)-1]:
        self[f"v{elem}"] = self.derive(elem,dt,opt=option)
        txt_v += f"v{elem}**2 +"
    txt_v = txt_v[:-1]
    for elem in xyzt[:len(xyzt)-1]:
        self[f"a{elem}"] = self.derive(f"v{elem}",dt,opt=option)
        txt_a += f"a{elem}**2 +"
    txt_a = txt_a[:-1]
    for elem in xyzt[:len(xyzt)-1]:
        txt_r += f"{elem}**2 +"
    txt_r = txt_r[:-1]
    self['r']=self.eval(f'({txt_r})**(1/2)')
    self["v"]=self.eval(f"({txt_v})**(1/2)")
    self["a"]=self.eval(f"({txt_a})**(1/2)")
    return self.head()

def derive(self,df,dt,opt=(1,0)):
    a,b = opt[0],opt[1]
    return (self[df].shift(-a)-self[df].shift(b))/(self[dt].shift(-a)-self[dt].shift(b))

def delta(self,df,opt=(1,0)):
    a,b = opt[0],opt[1]
    return (self[df].shift(-a)-self[df].shift(b))

def norme(self,*args):
    return np.sqrt(sum(self[arg]**2 for arg in args))

def del_columns(self,*y,**kwarg):
    y=list(y)
    y_elim = []
    display = kwarg.get('display',True)
    for elem in y:
        if elem not in list(self):
                if display:
                    print(f"... La colonne {elem} n'existe pas")
        else:
            y_elim.append(elem)
    if y_elim:
        self.drop(columns=y_elim,inplace=True)
    return self.head()

       
def nonlinear_regression(self,x,y,ym='--',func = lambda x,A,w,phy:A*np.sin(w*x+phy),ajout_col=False,tracer_courbe=False):
    if (x not in list(self.columns)) or (y not in list(self.columns)):
        print(f"la colonne {x} ou {y} n'existe pas dans votre dataframe")
        return self.columns
    idx = np.isfinite(self[x]) & np.isfinite(self[y])
    popt, _ = curve_fit(func, self[x][idx], self[y][idx])
    a = getsource(func)
    ok = False
    if "lambda" in a:
        ok = True
        i_deb = a.find(":",a.index("lambda"))+1
        i_fin = a.find(",",i_deb)
        a[i_deb:i_fin]
    list_arg = str(signature(func)).strip('(').strip(')').split(',')[1:]
    txt = ''
    for arg,val in zip(list_arg,popt):
        txt += f'{arg}={val}\n'
    if ym == '--':
        ym = y+'_mod'
    print(f"les paramètres de votre modèle {y}={a[i_deb:i_fin] if ok else ''} sont:\n{txt}")
    if tracer_courbe or ajout_col:
        self[ym] = func(self[x],*popt) 
    if tracer_courbe:
        self.scatter2D(x=x,y=[y,ym],style_tracer='-o')
    if not ajout_col:
        self.del_columns(ym)

def regression(self,x,y,ym='--',degre=1,ajout_col=False,tracer_courbe=True,**d):
    """ 
    Méthode/Fonction appliqué à une dataframe -> retourne une modélisation.
    
    Regression permet d'obtenir une modélisation entre 2 colonnes d'une dataframe df (courbe de tendance)
    * x (str): nom de la colonne correspondant à l'abcisse Ex : x='t(s)'
    * y (str): nom de la colonne correspondant à l'ordonnée, c'est la grandeur qui sera modélisée Ex: y='VX'
    * ym (str): nom de l'ordonnée modélisée Ex: y='VX_mod'
    * degre (0, 1 ou 2): est le depré du polynome (degre=1 pour une regression linéaire)
    * ajout_col (booleen): permet de creer une nouvelle colonne nommée ym dans la dataframe self (défaut ajout_col=False)
    * tracer_courbe : Si True, affiche la courbe y=f(x) et y_mod=f(x)
    
    Exemple : On veux modéliser par une relation linéaire la colonne 'V' en fonction de 'R' de la dataframe venus sans 
    creer une nouvelle colonne
    >venus.regression(x='V',y='R',degre=1) 
    
    résultat :
    meilleur modèle linéaire entre V et R est :
    V = 121558311055.52502 x R - 4.5721051617156645
    """
    if (x not in list(self.columns)) or (y not in list(self.columns)):
        print(f"la colonne {x} ou {y} n'existe pas dans votre dataframe")
        return self.columns
    degre = int(f_borne(degre,0,3))
    idx = np.isfinite(self[x]) & np.isfinite(self[y])
    resultat = np.polyfit(self[x][idx],self[y][idx],degre)
    p=np.poly1d(resultat,variable=x)
    if ym == '--':
        ym = y+'_mod'
    if tracer_courbe or ajout_col:
        self[ym] = np.poly1d(resultat)(self[x])
    print(f'meilleur modèle de degré {degre} entre {x} et {y} est :')
    print(y+'=')
    print(p)
    if tracer_courbe:
        self.scatter2D(x=x,y=[y,ym],**d)
    if not ajout_col:
        self.del_columns(ym)
    SPY.append({'method':'regression','x':x,'y':y,'degre':degre})


def f_read_wav(fichier,typ='tuple'):
    """
    fonction qui retourne la table des temps et celle des amplitudes (série numpy)
    Arguments:
        * fichier (wav) : chemin d'un fichier son
        * typ (str): si typ='tuple' la fonction retourne la tuple t,data ; si typ='dict' la fonction retourne un dictionnaire
            ayant pour clé 't','data' et 'sr' (samplerate)
    """
    rate,data = wavfile.read(fichier)
    nb_data = len(data)
    tmax = nb_data / rate
    t = np.linspace(0,tmax,nb_data)
    if typ == 'tuple':
        return t,data
    return {'t':t, 'data':data, 'sr':rate}

def f_plot(x, y, titre='Courbe sans titre!!!', xlabel='pas de nom', ylabel='pas de nom', quadril_x=0, quadril_y=0):
    """
    fonction qui retourne un graphique
    Arguments:
        * x (série numpy): nom de l'absisse
        * y (série numpy): nom des l'ordonnées (ex y=['y1','y2'])
        * titre (str): titre du tracé - Defaut 'Courbe sans titre!!!'
        * xlabel (str): nom de l'axe des absisses - Defaut 't(s)'
        * ylabel (str): nom de l'axe des ordonnées
    
    Exemple : Soient y1, y2 et t des séries numpy de taille identique, pour tracer y1=f(t) et y2=f(t)
            > plot(t,y=[y1,y2],titre='y1=f(t) et y2=f(t)')
    """
    if isinstance(y,np.ndarray):
        y=[y]
    courbes=[]
    for i,elem in enumerate(y):
        label=f'courbe n°{i}'
        if isinstance(elem,np.ndarray):
            courbes.append(go.Scatter(x=x, y=elem, mode='lines', name=label))
        else:
            print(f"L'argument n°{i} n'est pas une série numpy")
    layout = go.Layout(title= titre,xaxis= dict(nticks=quadril_x,title= xlabel),yaxis=dict(nticks=quadril_y,title= ylabel))
    fig = go.Figure(data=courbes, layout=layout)
    return iplot(fig,filename=titre)

def f_plot2(son=None,titre='Courbe sans titre!!!',xlabel='t(s)'):
    """
    fonction qui retourne un graphique
    Arguments:
        * son (dict): son audio ou liste de sons dont on veux tracer la (les) courbe(s)
        * titre (str): titre du tracé - Defaut 'Courbe sans titre!!!'
        * xlabel (str): nom de l'axe des absisses - Defaut 't(s)'
    
    Exemple : Soient son1 et son2 des sons, pour tracer leurs évolutions temporelles
            > f_plot2(son=[son1,son2],titre='un super titre',xlabel='t_s')
    """
    if not isinstance(son,list):
        son=[son]
    courbes=[]
    for i,elem in enumerate(son):
        if isinstance(elem,dict):
            if ('t' in elem) and ('data' in elem) and ('sr' in elem):
                courbes.append(go.Scatter(x=elem['t'],y=elem['data'],mode='lines',name=f'courbe n°{i}'))
    layout = go.Layout(title= titre,xaxis= dict(title= xlabel),yaxis=dict(title= 'signal'))
    fig = go.Figure(data=courbes, layout=layout)
    return iplot(fig,filename=titre)   
 
def f_plot_fft(son = None,x = None,y = None,titre='Courbe sans titre!!!'):
    """
    fonction qui retourne le graphique du spectre (fft) d'un son
    Arguments:
        * t (série numpy): nom de l'absisse
        * y (série numpy): nom de l'ordonnée
        * titre (str): titre du tracé - Defaut 'Courbe sans titre!!!'
        * son (dict) (oprionnel) : dictionnaire contenant t et data
    Exemple : Soient y et t des séries numpy de taille identique, pour tracer le spectre de y
            > plot_fft(t,y,titre='Spectre de y')
    """
    t = x
    data = y
    if isinstance(son,dict) and ('t' in son) and ('data' in son) and ('sr' in son):
        Y=np.abs(fft(son['data']))
        t=son['t']
        freq = fftfreq(len(son['data']), t[1] - t[0])
        data=None
    if isinstance(t,np.ndarray) and isinstance(data,np.ndarray):
        Y = np.abs(fft(data))
        freq = fftfreq(len(data), t[1] - t[0])
    f_plot(freq,Y,titre=titre,xlabel='f (Hz)')
    return {'freq':freq,'fft':Y}

def f_play_audio(son=None,t=None,data=None,file=None,rate=None):
    if isinstance(son,dict):
        return ip.display.Audio(data=son['data'],rate=son['sr'],autoplay=True)
    if isinstance(t,np.ndarray) and isinstance(data,np.ndarray):
        f_ech = len(t)/t.max()
        return ip.display.Audio(data=data,rate=f_ech,autoplay=True)
    if isinstance(file,str):
        return ip.display.Audio(filename=file,rate=rate,autoplay=True)

def f_delete_freq(son=None,freq=100,largeur=20):
    """
    Fonction qui retourne un son dont on aura retirer quelques fréquences
    Arguments:
        * son (dict clés 't','data','sr'): son à traiter
        * freq (float): fréquence ou liste de fréquences en Hertz (ex [150,140,552]) - Défaut 100
        * largeur (float): largeur de l intervalle de suppression en Hertz (ex si freq=150Hz et largeur=20Hz alors toutes les fréquences
        entre 130 et 170 seront supprimées) - Défaut 20
    """
    if not isinstance(freq,list):
        freq=[freq]
    dataFreq = fftshift(fft.fft(son['data']))
    sampleRate = son['sr']
    n = len(son['data'])
    w = largeur
    for f in freq:
        dataFreq[n/2 + n*f/sampleRate - w : n/2 + n*f/sampleRate + w] = 0
        dataFreq[n/2 - n*f/sampleRate - w : n/2 - n*f/sampleRate + w] = 0
    resultat = fft.ifft(fft.fftshift(dataFreq))
    return {'t':son['t'],'data':resultat,'sr':son['sr']}

def animation():
    anim = r"-/|\o"
    for i in range(20):
        sleep(0.05)
        print(anim[i % len(anim)],end='\r',flush=True)

def check_df(df,*t,opt='exist',suspense = False):
    """
    test si les arguments (str) sont des colonnes de la dataframe df
    """
    reussi = f"{Fore.GREEN} test réussi {Style.RESET_ALL}"
    echec = f"{Fore.RED} echec au test {Style.RESET_ALL}"
    resultat = True
    if len(t)==1 and isinstance(t[0],str) and '/' in t[0]:
        t = (t[0].strip('/')).split('/')
    for elem in t:
        if suspense: 
            animation()
        if opt == 'exist':
            if elem in list(df):
                print(f"La colonne '{elem}' est bien présente ==> {reussi}")
            elif elem == '__info':
                if not check_info(df):
                    print(f"Il n'y a pas d'information dans votre dataframe ==> {echec}")
                    print(f"Avertissement - dataframe sans nom\nEcrire ma_dataframe.info='quelque chose'")
                    resultat = False
                else:
                    print(f"Une info='{df.info}' est présente dans la dataframe ==> {reussi}")
            else:
                print(f"La colonne '{elem}' n'est pas présente ==> {echec}")
                resultat = False
        elif opt == 'noexist':
            if elem not in list(df):
                print(f"La colonne '{elem}' n'est pas présente ==> {reussi}")
            else:
                print(f"La colonne '{elem}' est encore présente ==> {echec}")
                resultat = False  
    sleep(1)
    if resultat:
        print(Back.GREEN+">>>Le test est réussi ..... vous pouvez continuer"+Style.RESET_ALL)
    else:
        print(Back.RED+">>>Malheureusement, le test est un echec, il faut recommencer"+Style.RESET_ALL)

def normalize(self,y,limit=(1,100),inplace=True):
    """
    normalise une la colonne y dans le limite de limit
    arguments:
        y (str) : nom de la colonne à normaliser
        limit (tuple ou list): limites de la normalisation
    Exemple:
        'x' est une colonne de la dataframe df variant de 0 à 527.
        pour normaliser la colonne 'x' et la faire varier de 0 à 100
        >df.normalize('x',limit=(0,100))
    """
    if isinstance(limit,(tuple,list)):
        limit = limit[:2]
    elif isinstance(limit,(int,float)):
        limit = [0,limit]
    elif isinstance(limit,str):
        if limit in self.columns:
            limit=[self[limit].dropna().min(),self[limit].dropna().max()]
        else:
            print(f"column {limit} does not exist ... limit fixed to [0,100]")
            limit = [0,100]
    else:
        limit = [0,100]
    y_min=self[y].dropna().min()
    y_max=self[y].dropna().max()
    if inplace:
        self[y] = self[y].apply(lambda y: (y- y_min)/(y_max-y_min)*(max(limit)-min(limit))+min(limit))
    else:
        return self[y].apply(lambda y: (y- y_min)/(y_max-y_min)*(max(limit)-min(limit))+min(limit))
        
def check_op(self,y,exp,suspense=True):
    """
    vérifie que colonne la colonne y est définie par l'expression mathémétique exp
    Exemple: soit df une dataframe possédant 't' comme colonne de temps et 'x' vérifiant
        x = 3*t-9
        >df.check_op('x','6iHiéA2|ç') retourne :
            La fonction x=3*t-9 convient ->  Le test est réussi
    """
    expd = f_crypt(exp,SPY_KEY,"decrypt")
    #print(expd)
    try:
        self[f"__eval_{y}"] = self.eval(expd)
        resultat = np.isclose(self[y].fillna(0),self[f"__eval_{y}"].fillna(0))
        #☻print(resultat)
    except:
        print(f"Rien ne va, votre crypto est mauvais\n{expd}")
        return None
    if suspense:
        animation()
    if resultat.all():
        print(f"La fonction {y}={expd} convient -> {Back.GREEN} Le test est réussi"+Style.RESET_ALL)
    else:
        print(f"La fonction qui définie {y} ne convient pas -> {Back.RED} echec"+Style.RESET_ALL)
    self.del_columns(f"__eval_{y}")

def check_info(df):
    return False if "bound method DataFrame.info" in str(df.info) else True
    
def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return  same


def check_graph(txt_c,level=0,suspense=True):
    secret = f_crypt(txt_c,SPY_KEY,"decrypt")
    try:
        dd = eval(secret)
        t = dd['method']
    except:
        print(f"Rien ne va, votre crypto est mauvais\n {secret}")
        return None
    same = dict_compare(dd,SPY[-1-level])
    if suspense:
        animation()
    if len(same) == len(dd):
        print(f"Votre {t} {dd} est correct ==> {Fore.GREEN} test réussi {Style.RESET_ALL}")
    else:
        print(f"Votre {t} n'est pas correct ==> {Fore.RED} echec au test {Style.RESET_ALL}")
            
def reset_spy():
    SPY = []

def get_spy():
    return SPY

def check_func(func,check_value,*arg):
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
        if np.isclose(resultat,check_value).all():
            res+=f"==> {Fore.GREEN}Test réussi{Style.RESET_ALL}"
        else:
            res+=f"==> {Fore.RED}Echec ... revoir votre fonction{Style.RESET_ALL}"  
    else:
        if resultat == check_value:
            res+=f"==> {Fore.GREEN}Test réussi{Style.RESET_ALL}"
        else:
            res+=f"==> {Fore.RED}Echec ... revoir votre fonction{Style.RESET_ALL}"
    print(res)

def give_me_crypto(levels=0):
    if not isinstance(levels,(list,tuple)):
        levels = [levels]
    for level in levels:
        d = str(SPY[-level-1])
        dc = f_crypt(d,SPY_KEY,"crypt")
        print(f"level={level} .. {d}..'{dc}'")

def f_crypt(y,key,option="crypt"):
    clair = sorted(r"ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz0123456789²&é'(-è_çà)#{[|^@]}?,;.:/!§*+µ$£%=",key=str.lower)
    clair = "".join(clair)
    crypte = "".join([clair[(key+i)%len(clair)] for i,_ in enumerate(clair)])
    if option == "crypt":
        return "".join([crypte[clair.find(i)] for i in y])
    else:
        clair = "".join([crypte[(-key+i)%len(crypte)] for i,_ in enumerate(crypte)])
        return "".join([clair[crypte.find(i)] for i in y])
       
def f_input(txt='',output='float',limit=None,choice=None,sep=','):
    #sep_list = list(r"-,;:_/\")
    sep_list = "/\\,;:-"
    if sep not in sep_list:
        print(f"Err: delimiter sep='{sep}' must be in ({sep_list})")
        return None
    output_list = ["float","float_list","int","int_float","str"]
    if output not in output_list:
        print(f"Err: argument output={output} must be in {output_list}")
        return None
    if isinstance(choice,(tuple,list)):
        affichage = f"choice={choice}"
        option = 'choice'
    elif isinstance(limit,(tuple,list)):
        limit = limit[:2]
        affichage = f"limit={limit}"
        option = 'limit'
    else:
        affichage = '*'
        option = ''
    while True:
        res = input(f"{txt} [{Fore.GREEN}{output}{Style.RESET_ALL}][{Fore.RED}{affichage}{Style.RESET_ALL}]>")
        if res.lower() == "_exit":
            break
        if output == 'float':
            try:
                resultat = float(res)
                if option == 'limit':
                    if (max(limit)>=resultat>=min(limit)):
                        break
                    else:
                        print(f"Err: input {resultat} out of range {limit}")
                elif option == 'choice':
                    if resultat in choice:
                        break
                    else:
                        print(f"Err: input {resultat} not in choice {choice}")
                else:
                    break
            except:
                print("Result isnt float")
        elif output == 'int':
            try:
                resultat = int(res)
                if option == 'limit':
                    if (max(limit)>=resultat>=min(limit)):
                        break
                    else:
                        print(f"Err: input {resultat} out of range {limit}")
                elif option == 'choice':
                    if resultat in choice:
                        break
                    else:
                        print(f"Err: input {resultat} not in choice {choice}")
                else:
                    break
            except:
                print("Err:Result isnt integer") 
        elif output == "float_list":
            try:
                resultat = [float(i) for i in res.split(sep)]
                if option == 'limit':
                    if all([(max(limit)>=i>=min(limit)) for i in resultat]):
                        break
                    else:
                        print(f"Err: input {resultat} out of range {limit}")
                elif option=='choice':
                    if all([i in choice for i in resultat]):
                        break
                    else:
                        print(f"Err: input {resultat} not in choice {choice}")
                else:
                    break
            except:
                print(f"les valeurs données ne sont pas des floats")
        elif output == "int_list":
            try:
                resultat = [int(i) for i in res.split(sep)]
                if option == 'limit':
                    if all([(max(limit)>=i>=min(limit)) for i in resultat]):
                        break
                    else:
                        print(f"Err: input {resultat} out of range {limit}")
                elif option == 'choice':
                    if all([i in choice for i in resultat]):
                        break
                    else:
                        print(f"Err: input {resultat} not in choice {choice}")
                else:
                    break
            except:
                print(f"il y a un probleme")
        elif output == 'str':
            resultat = res
            if option == 'choice':
                if resultat in choice:
                    break
                else:
                    print(f"Err: input {resultat} not in choice {choice}")
            else:
                break
    return resultat

def f_input_list(txt, limit=None, sep=','):
    result = []
    result_txt = f_input(txt,opt='str')
    try:
        result = [float(i) for i in result_txt.split(sep)]
    except:
        print(f'il y a un problème dans vos valeurs :\n{result_txt}')
        return None
    if limit != None:
        result = [f_borne(i,limit[0],limit[1]) for i in result]    
    return result
          
def f_borne(x,xmin,xmax):
    return max(xmin,min(xmax,x))             

def f_sin(x):
    """
    calcule le sinus de x ou x est en degré
    """
    x = x/180*np.pi
    return np.sin(x)

def f_cos(x):
    """
    calcule le sinus de x ou x est en degré
    """
    x = x/180*np.pi
    return np.cos(x)
    
def f_arcsin(x):
    """
    calcule le sinus d'arc de x et retourne la valeur en degré
    """
    return np.arcsin(x)/np.pi*180

def draw_refrac(i1,i2,n1='',n2='',d = 10.0):
    trace0 = go.Scatter(x=[0,d/2,-d/2],y=[0,d/2,-d/2],text=['normale',f"n1={n1:.4f}",f"n2={n2:.4f}"],mode='text')
    data = [trace0]
    layout = {'xaxis': {'range': [-d, d],'constrain':'domain'},'yaxis': {'range': [-d, d],'scaleanchor':'x'},
    'shapes': [ {'type': 'line','x0': 0,'y0': 0,'x1': -d*f_sin(i1),'y1': d*f_cos(i1),
                 'line': {'color': 'rgb(55, 128, 191)','width': 3,}},
                {'type': 'line','x0': 0,'y0': 0,'x1': d*f_sin(i2),'y1': -d*f_cos(i2),
                 'line': {'color': 'rgb(50, 171, 96)','width': 4}},
                 {'type':'rect','x0':-d,'y0':-d,'x1':d,'y1':0,'fillcolor': 'rgba(128, 0, 128, 0.7)','opacity':0.5}]}
    fig = {'data': data,'layout': layout}
    return iplot(fig, filename='shapes-lines')
 
def f_test(func,result,*arg,ecart = 1e-4):
    r = func(*arg)
    if isinstance(r,float):
        resul_test =  np.abs(r-result)/r<ecart
    if isinstance(r,int):
        resul_test = (r == result)
    if resul_test:
        print(f"{Fore.GREEN}le test de la fonction {func.__name__} à réussi .. {Style.RESET_ALL}")
    else:
        print((f"{Fore.RED}le test de la fonction {func.__name__} est un echec{Style.RESET_ALL} .. il faut recommencer "))

def y_to_str(y):
    if isinstance(y,str) and '/' in y:
        return '/'.join(sorted(y.split('/'),key=str.lower))
    if isinstance(y,str) and '/' not in y:
        return y
    if isinstance(y,list):
        return '/'.join(sorted(y,key=str.lower))
    return None

def format_to_valid_y(y):
    if isinstance(y,str) and '/' in y:
        return y.split('/')
    if isinstance(y,str) and '/' not in y:
        return [y]
    if isinstance(y,list):
        return y
    if isinstance(y,tuple):
        return list(y)

def Nasa_horizons_query(id='3',id_type='majorbody', origin='@sun',epochs=dict(start='2016-10-01',stop='2017-10-02',step='10m'),**d):
    print("connect to NASA JPL Horizons ...\r",end='\r')
    obj = Horizons(id=id,id_type=id_type,location=origin,epochs=epochs).vectors().to_pandas()
    astre = obj['targetname'][0]
    print(f"query of {astre} ..... start={epochs['start']}..end={epochs['stop']}.... finish")
    if not d.get('keep_date',False):
        obj.del_columns('datetime_str',display=False)
    else:
        obj['datetime_str']=pd.to_datetime(obj['datetime_str'],format= 'A.D. %Y-%b-%d %H:%M:%S.0000')
        obj.rename({'datetime_str': 'date'}, axis=1, inplace=True)
    obj.del_columns('targetname', 'H', 'G',
   'vx', 'vy', 'vz', 'lighttime', 'range', 'range_rate',display=False)
    delta_t = (obj['datetime_jd'][1]-obj['datetime_jd'][0])*24*3600
    obj.del_columns('datetime_jd',display=False)
    obj['t'] = delta_t*obj.index
    #convert to AU
    obj['x'] = 1.496e+11*obj['x']
    obj['y'] = 1.496e+11*obj['y']
    obj['z'] = 1.496e+11*obj['z']
    obj.info = f"{astre}"
    return obj

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

pd.DataFrame.delta = delta
pd.DataFrame.check_op = check_op
pd.DataFrame.vector = vector
pd.DataFrame.vector3D = vector3D
pd.DataFrame.regression = regression
pd.DataFrame.scatter3D = scatter3D
pd.DataFrame.scatter2D = scatter2D
pd.DataFrame.display_U = display_U
pd.DataFrame.derive = derive
pd.DataFrame.norme = norme
pd.DataFrame.del_columns = del_columns
pd.DataFrame.nonlinear_regression = nonlinear_regression
pd.DataFrame.normalize = normalize
pd.DataFrame.scatterPolar = scatterPolar
pd.DataFrame.histogram = histogram
pd.DataFrame.draw_vectors = draw_vectors
pd.DataFrame.get_kinematic = get_kinematic