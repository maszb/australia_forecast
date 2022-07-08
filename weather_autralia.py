import streamlit as st
import pandas as pd
import folium
import missingno as msno
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, GridSearchCV
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

old_data = pd.read_csv("datasets//weatherAUS.csv")

df=pd.read_csv("datasets//WeatherAUSRegionTAS.csv",sep=';')
df2=pd.read_csv("datasets//WeatherAUSRegionNIS.csv",sep=';')
df_villes=pd.read_csv('datasets//AUSLONLAT.csv',sep=';')
villesref=pd.read_csv("datasets//VilleRef.csv",sep=";")
villesref.rename(columns = {'Ville':'city'}, inplace = True)
dfW=pd.concat([df,df2],ignore_index=True)
dfW=df.drop('Date',axis=1)


australian_clean=pd.read_csv("datasets//WeatherAUSFinal.csv",sep=";")
australian_clean.head(5)
a_clean_f = australian_clean[australian_clean['Date']=='2017-05-12']

lista = []
for i in range (0,a_clean_f.shape[0]):
    lista.append(i)

a_clean_f.set_index([lista])

a_clean_f.replace('No', 0, inplace=True)
a_clean_f.replace('Yes', 1, inplace=True)
a_clean_f.head(5)
a_clean_f.set_index([lista], inplace=True)
a_clean_f.tail(5)

a_clean_f.rename(columns = {'Location':'city'}, inplace = True)
df_cd = pd.merge(a_clean_f, villesref, how='inner')
df_cd_1 = pd.merge(df_cd, df_villes)
df_cd_1.head(5)

df_villes_ref={}
for idx,lige in villesref.iterrows():
    df_villes_ref[lige['city']]= lige['Radar']

m=folium.Map(location=[df_cd_1[df_cd_1['city']=='Sydney']['lat'],
                       df_cd_1[df_cd_1['city']=='Sydney']['lng']],
             zoom_start=5)

url="E:\Projet\{}".format
urlimage=url("IconSun.png")
urlimageradar=url("IconSun2.png")



def ReplaceRain(dataframe):
    dataframe['RainToday'] = dataframe['RainToday'].replace (to_replace = ['Yes','No'], value = [1,0])
    dataframe['RainTomorrow'] = dataframe['RainTomorrow'].replace (to_replace = ['Yes','No'], value = [1,0])

def GetDummies(dataframe, variable):
    dummies = pd.get_dummies(dataframe[variable])
    dataframe = pd.concat([dataframe,dummies], axis = 1)
    dataframe=dataframe.drop(variable, axis = 1)
    return dataframe

def Standardise(dataframe,colonnes):
    scaler = StandardScaler()
    dataframe[colonnes]= scaler.fit_transform(dataframe[colonnes])
    return dataframe[colonnes]

lencoders = {}
for col in dfW.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    dfW[col] = lencoders[col].fit_transform(dfW[col])

dfW=dfW.drop('week',axis=1)

# Nom du projet 
st.title("Australia Forecast")

pages = ["Introduction", "Datasets", "Exploration des données et visualisation", "Modélisation", "Prédiction", "Conclusions"]

page = st.sidebar.radio("Menu", options = pages)

if page == pages[0]:
    st.image("images//image001.jpg")
    st.markdown("""
Cet ensemble de données contient environ 10 ans d'observations météorologiques quotidiennes provenant de nombreux endroits en Australie.
Il y a donc différentes visualisations intéressantes possibles.
Le premier objectif serait de prédire la variable cible : RainTomorrow. Elle signifie : a-t-il plu le jour suivant, oui ou non ? Cette colonne est Oui si la pluie pour ce jour était de 1mm ou plus.
De même pour des prédictions de vent ou température.
Dans un second temps, on pourra effectuer des prédictions à long terme, en utilisant des techniques mathématiques d’analyse de séries temporelles, et/ou des réseaux de neurones récurrents.

Data 
-	https://www.kaggle.com/jsphyg/weather-dataset-rattle-package

Benchmark/ Bibliographie/Source:
-	Les observations proviennent de nombreuses stations météorologiques. Les observations quotidiennes sont disponibles sur http://www.bom.gov.au/climate/data.
-	Un exemple des dernières observations météorologiques à Canberra : http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml
-	Définitions adaptées de http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml
-	Source des données : http://www.bom.gov.au/climate/dwo/ et http://www.bom.gov.au/climate/data.
    """)

if page == pages[1]:
    st.markdown("""
    Nous avons travaillé sur un fichier csv (weatherAUS.csv). Ce dataset fournit des données météorologiques de 49 villes d’Australie entre 2009 et 2017. Le jeu de données initial comportait 145 460 lignes et 23 colonnes. 
    La variable à prédire est RainTomorrow. Elle est renseignée à Yes ou No.

Les années 2007 2008 et 2017 sont très incomplètes. Les autres années sont complètes bien qu’il puisse manquer quelques jours dans certains mois et certaines villes.
Le manque de données se concentre beaucoup sur Sunshine et Evaporation ou les variables Pressure ce qui nous a posé problème pour identifier une méthode fiable de comblement des Nan.

Certaines variables marchent par couple :  données à 9 heures du matin et une autre à 3 heures de l’après-midi. C’est le cas de :
-   Cloud
-   Pressure
-   Windir
-   WinSpeed
-   Temp

    """)
    st.dataframe(old_data)
    
    resumer = st.checkbox('Afficher le résumé')
   

    if resumer:
        st.dataframe(old_data.describe())

    dim = st.checkbox('Afficher les différentes villes')
    if dim:
        st.markdown("""
        Les villes ne sont pas équitablement réparties sur le territoire. Le centre du pays est clairement sous représenté.
        """)
        st.image("images//carte_villes.PNG")

if page == pages[2]:
    st.header("Analyse des données")
    st.markdown("""
    Pour mener à bien un projet de data science, nous procédons toujours à une première étape qui est 
    l'analyse des données. Cette étape constitue un enjeu majeur pour une annalyse approfondie des données brutes pour rechercher des modèles, des tendances et des mesures dans un ensemble de données.
    
    Choix des variables :\n
    Description des variables du dataset  
    """)
    st.image("images//variable1.PNG")
    st.image("images//variable2.PNG")

    st.header("Analyse statistique du dataset")
    st.markdown("""
    Après la phase de préparation des données, une première analyse statistique de la base de données a été réalisée en utilisant la méthode describe. Le résultat est présenté ci-dessous :    """)
    old_data=pd.read_csv("datasets//WeatherAUSFinal.csv",sep=";")
    st.write(old_data.describe())
    
    st.markdown("""
    Sur une période de 8 ans, le tableau montre que les valeurs moyennes des températures minimales et maximales sont respectivement de 12 et 23 degrés Celsius, les valeurs moyennes de pression enregistrées le matin et l'après-midi sont de 1017 et 1015 et la quantité moyenne de pluie est de 2,34 mm. Bien que la plupart des stations météorologiques soient situées sur la côte, où les précipitations sont plus fréquentes que dans la zone centrale, l'Australie est un pays principalement aride et, dans l'ensemble, le pays n'est pas caractérisé par de fortes précipitations. Les quartiles sont également indiqués dans le tableau. Une façon plus simple d'analyser les outils est de représenter graphiquement les données de la trame de données dans ce que l'on appelle le graphique à moustaches. Le graphique est présenté ci-dessous.""")   

    st.image("images/boxplot.png")
    st.markdown("""
    En analysant le graphique, on constate que pour les variables Pluie et Évaporation, les valeurs extrêmes sont situées au-dessus du troisième quartile. Conformément au tableau ci-dessus, nous pouvons déduire que ce n'est qu'à certaines périodes de l'année que la quantité de précipitations est supérieure à la moyenne. Il en va de même pour la variable Evaporation et la variable Vitesse du vent. Contrairement à ces variables, la variable humidité enregistrée à 9h du matin est caractérisée par un faible nombre d'outils, qui se situent en dessous du premier quartile. En entrant dans le détail, il est intéressant de noter où les plus fortes précipitations ont été enregistrées en huit ans d'échantillonnage. Le graphique cumulatif de la variable "précipitations" pour les différentes stations météorologiques et années est présenté ci-dessous.
    """)

    st.markdown("""
    Le graphique montre une relation linéaire étroite entre les deux variables considérées.
    """)
    sns.pairplot(data=dfW, vars=('Pressure3pm','Pressure9am'), hue='RainTomorrow' )
    st.pyplot(plt.show())

    st.markdown("""
    Le graphique ci-après montre la température maximale en dégré celcius par rapport à la température minimale enregistrée en 24 heures jusqu'à 9 heures du matin en dégré celcius.
    """)
    sns.pairplot(data=dfW, vars=('MinTemp','MaxTemp'), hue='RainTomorrow' )
    st.pyplot(plt.show())
    st.markdown("""
    Le graphique ci-dessus montre une relation linéaire entre les variables prises en considération. La température minimale est d'environ -8 dégré celcius et la maximale est d'environ 48 dégré celcius. La carte satellite de l'Australie montre que, dans les régions du sud-est et du sud-ouest, le climat est plus tempéré, rendant l'air propice à l'implantation humaine ; c'est dans ces régions que l'on trouve les grandes villes australiennes telles que Sydney, Perth ou Melbourne. Étant donné que l'Australie est située dans l'hémisphère sud, la différence de température d'une région à l'autre est moins prononcée, mais en raison de son immensité et de son climat essentiellement aride, les écarts de température peuvent être importants.
    """)

    st.header("Tests de dépendances des variables")
    st.markdown("""
    Pour tester la dépendance entre les variables, nous avons utilisé le test de CHI2 et V de Cramer et le Test ANOVA.
    Pour 5 variables en lien avec RainTomorrow, nous avons obetenu les résultats suivants:
    """)
    st.markdown("""
    df['WindGustDir'], df['RainTomorrow'] pas de forte corrélation\n
    df['WindGustSpeed'], df['RainTomorrow'] pas de forte correlation\n
    df['Humidity3pm'], df['RainTomorrow'] corrélation Moyenne\n
    df['Sunshine'], df['RainTomorrow'] corrélation Moyenne\n
    df['RainToday'], df['RainTomorrow'] corrélation Moyenne\n
    """)

    st.header("Selection des variables à abandonner")
    st.markdown("""
    MaxTemp est la température maximale de la journée. Elle est très proche de Temp9am. Nous avons donc décidé de dropper Temp9am
    MinTemp est la température prise à 9 heures. Elle correspond à Temp9am. Nous avons donc décidé de dropper MinTemp.
    Le même constat a été fait pour les variables WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm.
    En revanche Pressure9am et Pressure3pm n’ont pas d’équivalence, nous les avons donc garder toutes les deux.
    Nous avons choisi de ne pas garder les variables Cloud9am et Cloud3pm.\n

    Nous nous séparons de Katherine et Launceston. Il y a trop peu de données exploitables et pas de ville radar pertinentes pour ces 2 villes.
    Par ailleurs nous nous séparons aussi des années incomplètes (qui ne s’étalent pas du 1er janvier au 31 décembre).
    Enfin les lignes ou RainToday et RainTommorrow sont à null sont également supprimées.

    """)
    st.image("images//variable3.PNG")

    st.header("Ajout des nouvelles variables")
    st.markdown("""
    Comme nous devrons travailler à la journée, nous avons décidé de découper la date en 3 variables supplémentaires : le jour, le mois et l’année.
    """)

    st.header("Identification et répartition des NaNs")
    st.markdown("""
    Nous avons travaillé avec le module MissingNo et nous avons analysé ville par ville qu’elles étaient les valeurs absentes et dans quelles proportions. 
    Les valeurs manquantes se concentrent beaucoup sur 2 couples de valeurs : Evaporation et Sunshine et Pressure9am et Pressure3pm. Il y a un lien fort entre Evaporation et Sunshine quand il s’agit de valeurs manquantes. La même chose se constate pour les deux variables Pressure. Certaines villes n’ont aucune valeur pour le couple de variables !\n
    Vous trouverez ci-après quelques figures pour vous donner une idée de la répartition des données manquantes par ville.
    """)
    st.image("images//msn1.png")
    st.image("images//msn2.png")
    st.image("images//msn3.png")
    st.image("images//msn4.png")
    st.image("images//msn5.png")
    st.image("images//msn6.png")
    st.image("images//msn7.png")
    st.image("images//msn8.png")
    st.image("images//msn10.png")

    st.header("Traitement des valeurs manquantes")
    st.markdown("""
    Plusieurs méthodes s’offraient à nous. Remplir avec la méthode bfill(), ffill(), le mode() ou le mean(). Nous avons opté pour une méthode plus réaliste et travaillé sur la base de la proximité des villes et le fait que le temps soit similaire. 

    Nous avons parcouru les sites donnés dans la fiche projet et nous avons principalement exploité le site météo du gouvernement australien  (http://www.bom.gov.au) d’où sont tirées les données météorologiques du pays. 

    Le principe est de fonctionner sur la notion de radar : zone de quelques kilomètres autour d’une ville où l’estimation du temps s’applique aux villes environnantes. 
    Nous avons repris cette notion de ville radar pour notre projet et l’avons appliqué à notre jeu de données (page suivante exemple avec Moree).
    """)
    

if page == pages[3]: 
    pass

if page == pages[4]:
    st.header("Conclusions")
    st.markdown("""
    Ce projet de master s'est concentré sur la prédiction de la variable rainTomorrow, et a permis de mettre en pratique les concepts étudiés dans le cadre de la formation Data Scientest. Afin de prédire la variable, des protocoles ont été développés sur la base de l'état de l'art en matière de calcul. Afin de développer ces protocoles, un traitement du jeu de données a d'abord été effectué, suivi d'une analyse statistique et graphique, permettant d'obtenir une vue d'ensemble.\n\n
    Comme les prévisions météorologiques peuvent suivre des cycles dans le temps, une étude des séries temporelles a été entreprise. Cependant, il faut savoir que le manque de mémoire de nos ordinateurs fait planter le traitement. Cela peut être imputable à la taille de l'ensemble de données ; ce résultat a également été obtenu dans le cas d'un ensemble de données réduit (3 à 5 ans). 
    """)
    st.image("images//conclusion1.png")
    st.markdown("""
    Les meilleurs hyperparamètres varient parfois selon les régions administratives. En revanche, les hyperparamètres de la méthode de la forêt aléatoire sont les mêmes pour toutes les régions. De plus, cette méthode s'est avérée être la plus rapide (en termes de temps réel) par rapport aux deux autres méthodes.
    """)
    st.image("images//conclusion2.png")
    st.markdown("""
    En conclusion, notre étude nous a permis à la fois de déterminer la variable catégorielle en question avec une bonne précision et de déterminer les hyperparamètres optimaux. Ce travail jette les bases des prédictions et des améliorations futures du modèle. En effet, la méthode de régression linéaire sur les ensembles de données VIC et NSW devrait également être incluse dans les perspectives futures. \n\n
    Enfin, il faut souligner que l'utilisation de la bibliothèque missingo montre que l'ensemble de la base de données est caractérisé par un nombre élevé de valeurs manquantes. Par conséquent, il serait intéressant de tester la transférabilité du modèle sur un ensemble de données moins manipulées afin de tester ce protocole prédictif et de mettre en évidence ses performances.\n\n
    Nous avons tenté de prédire la vitesse du vent avec Ridge et Lasso mais les résultats n’ont pas été pas concluants.

    """)
