import streamlit as st

from streamlit_folium import st_folium

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

result_NSW = pd.read_csv("datasets//resultofNSW.csv",sep=";")
result_WEA = pd.read_csv("datasets//resultofWEA.csv",sep=";")
australian_clean=pd.read_csv("datasets//WeatherAUSFinal.csv",sep=";")


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



pages = ["Introduction", "Datasets", "Exploration des données et visualisation", "Algorithme de traitement des NaN", "Modélisation", "Conclusions","Restitution graphique des prédictions", "Entrainez nos modèles ! ","Lancez une prédiction !"]



page = st.sidebar.radio("Menu", options = pages)



if page == pages[0]:

    st.image("images//image001.jpg")
    

    st.markdown("""


Projet présenté par Zeina ACHKAR, Sandrine ASSERAF, Bernardino TIRRI et Magarh TSILOUONI\n
Promotion Mai 2022 DataScientest mode Bootcamp\n\n
     
       
       
 """)
    st.markdown("")
    st.markdown("")
    st.markdown("""
\n\nLe dataset contient environ 10 ans d'observations météorologiques quotidiennes provenant de nombreux endroits en Australie.

Il y a donc différentes visualisations intéressantes possibles.

Le premier objectif serait de prédire la variable cible : RainTomorrow. Elle signifie : a-t-il plu le jour suivant, oui ou non ? Cette colonne est Oui si la pluie pour ce jour était de 1mm ou plus.

Dans un second temps, on pourra effectuer des prédictions à long terme, en utilisant des techniques mathématiques d’analyse de séries temporelles.



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

Le manque de données se concentre beaucoup sur Sunshine et Evaporation ou les variables Pressure ce qui nous a posé problème pour identifier une méthode fiable de comblement des NaN.



Certaines variables marchent par couple :  données à 9 heures du matin et une autre à 3 heures de l’après-midi. C’est le cas de :

-   Cloud

-   Pressure

-   Windir

-   WinSpeed

-   Temp



    """)

    st.dataframe(old_data)

    st.markdown("")
    st.markdown("")

    resumer = st.checkbox('Afficher le résumé')

   



    if resumer:

        st.dataframe(old_data.describe())



    st.markdown("""

        Les villes ne sont pas équitablement réparties sur le territoire. Le centre du pays est clairement sous représenté.

        """)

    st.image("images//carte_villes.png")



if page == pages[2]:

    st.header("Analyse des données")

    st.markdown("""

    Pour mener à bien un projet de data science, nous procédons toujours à une première étape qui est l'analyse des données. Cette étape constitue un enjeu majeur pour une analyse approfondie des données brutes pour rechercher des modèles, des tendances et des mesures dans un ensemble de données.

    

    Description des variables du dataset  

    """)

    st.image("images//variable1.png")

    st.image("images//variable2.png")
    st.markdown("")
    st.markdown("")



    st.header("Analyse statistique du dataset")

    st.markdown("""

    Après la phase de préparation des données, une première analyse statistique de la base de données a été réalisée en utilisant la méthode describe. Le résultat est présenté ci-dessous :    """)

    st.write(old_data.describe())

    

    st.markdown("""

    Sur une période de 8 ans, le tableau montre que les valeurs moyennes des températures minimales et maximales sont respectivement de 12 et 23 degrés Celsius, les valeurs moyennes de pression enregistrées le matin et l'après-midi sont de 1017 et 1015 et la quantité moyenne de pluie est de 2,34 mm. Bien que la plupart des stations météorologiques soient situées sur la côte, où les précipitations sont plus fréquentes que dans la zone centrale, l'Australie est un pays principalement aride et, dans l'ensemble, le pays n'est pas caractérisé par de fortes précipitations. Les quartiles sont également indiqués dans le tableau. Une façon plus simple d'analyser les outils est de représenter graphiquement les données de la trame de données dans ce que l'on appelle le graphique à moustaches. Le graphique est présenté ci-dessous.""")   



    st.image("images/boxplot.png")

    st.markdown("""

    En analysant le graphique, on constate que pour les variables Pluie et Évaporation, les valeurs extrêmes sont situées au-dessus du troisième quartile. Conformément au tableau ci-dessus, nous pouvons déduire que ce n'est qu'à certaines périodes de l'année que la quantité de précipitations est supérieure à la moyenne. Il en va de même pour la variable Evaporation et la variable Vitesse du vent. Contrairement à ces variables, la variable humidité enregistrée à 9h du matin est caractérisée par un faible nombre d'outils, qui se situent en dessous du premier quartile. En entrant dans le détail, il est intéressant de noter où les plus fortes précipitations ont été enregistrées en huit ans d'échantillonnage. Le graphique cumulatif de la variable "précipitations" pour les différentes stations météorologiques et années est présenté ci-dessous.

    """)

    st.image("images//region.png")

    st.markdown("""

    Les graphiques montrent que les villes caractérisées par de fortes précipitations sont Cairns, Darwin, Coffs Harbour, Gold Coast et enfin Wollongong. Ces stations météorologiques sont situées respectivement dans les régions NSW, QUE, VIC, WAU et NTE. Si l'on prend la variable RainFall comme indicateur, les années caractérisées par de fortes précipitations sont 2010 et 2011. Les graphiques cumulatifs de la variable catégorielle RainToday et RainTomorrow sont présentés ci-dessous.

    """)

    st.image("images//region2.png")



    st.markdown("""

    Le graphique ci-dessous montre une relation linéaire étroite entre les deux variables considérées.

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
    
    st.markdown("")
    st.markdown("")



    st.header("Tests de dépendances des variables")

    st.subheader("CHI2 et V de Cramer")

    st.markdown("""

    Nous avons fait de nombreux tests afin d’évaluer les dépendances entre variables. Le code suivant a été exécuté pour 5 variables en lien avec RainTomorrow. Nous ne mettrons pas le détail d’exécution de chaque variable, mais juste les résultats.

    """)

    st.markdown("""

    df['WindGustDir'], df['RainTomorrow'] pas de forte corrélation\n

    df['WindGustSpeed'], df['RainTomorrow'] pas de forte correlation\n

    df['Humidity3pm'], df['RainTomorrow'] corrélation Moyenne\n

    df['Sunshine'], df['RainTomorrow'] corrélation Moyenne\n

    df['RainToday'], df['RainTomorrow'] corrélation Moyenne\n

    """)

    st.subheader("Tests ANOVA")

    st.markdown("""

    Pour les variables comprenant de nombreuses modalités nous avons effectué des tests Anova :

    MinTemp, MaxTemp Evaporation et Pressure3pm



    Si la p_value (PR>F) est <5%, on rejette l'hypothèse H0 qui dit que les 2 variables X et Y sont indépendantes et on déduit du test que X a un effet statistique significatif sur la variable cible.



    Pressure 3pm a un effet significatif sur RainTomorrow

    MinTemp a un effet statistique significatif sur RainTomorrow

    MaxTemp a un effet statistique significatif sur RainTomorrow

    Evaporation a un effet statistique significatif sur RainTomorrow



    Nous avons aussi fait les tests HI2 et Anova entre Cloud et Humidity, pour jauger la corrélation entre ces variables. Les tests ont montré un effet statistique significatif entre Cloud9am et Humidity9am et entre Cloud3pm et Humidity3pm.\n
    Pour ces raisons de "non indépendance" des variables Cloud9am et Humidity9am,  puis de Cloud3pm et Humidity3pm, et sachant que Cloud9am et Cloud3pm comportent près de 60 000 valeurs de NaN chacune (au niveau du dataset total qui comportent près de 142 000 lignes), nous avons décidé de nous séparer de ces deux variables Cloud9am et Cloud3pm.\n\n



    """)
    st.markdown("")
    st.markdown("")



    st.header("Sélection des meilleures features")

    st.markdown("""

    Nous avons lancé un SlectFromModel afin d’identifier les meilleures features pour nos prédictions sur le jeu de données Tasmanie et Norfolk Island. Voici le résultat :

    """)

    st.image("images//SelectBest.png")

    st.markdown("""

    

    Nous avons choisi de conserver toutes les variables car il y en a assez peu dans le dataset, mais nous aurons au moins fait cet exercice qui confirme assez les tests de Cramer fait également.



    """)



    st.header("Selection des variables et des données à abandonner")

    st.markdown("""

    MaxTemp est la température maximale de la journée. MinTemp est la température minimale de la journée. 

    Temp9am est la température mesurée à 9 heures du matin. Temp3pm est la température mesurée à 15 h.\n 
    
    L'information des températures à 9h du matin et à 15h de l'après-midi semblent redondantes avec les infos des températures extrêmes de la journée (qui semblent plus importantes). 
    Nous avons choisi de garder MaxTemp et MinTemp et d'abandonner Temp9am et Temp3pm.\n

    Nous constatons la même chose pour la Direction du Vent (WindGustDir, WindDir9am, WindDir3pm), et la Vitesse du Vent (WindGustSpeed, WindSpeed9am, WindSpeed3pm), où nous avons les variables WindGustDir  désignant la direction du vent le plus fort de la journée, et WindGustSpeed la vitesse du vent le plus fort de la journée, qui semblent donner des infos plus importantes, que les variables regroupant les observations du vent à 9h du matin et 15 de l'après-midi. 
    Pour cette raison, nous avons choisi de garder WindGustDir et WindGustSpeed et d'abandonner WindDir9am, WindDir3pm,  WindSpeed9am, WindSpeed3pm.\n
    
    Les variables Pressure9am et  Pressure3pm donnent les niveaux de la Pression Atmosphérique à 9h du matin et à 15h de l'après-midi.\n 
    
    Les variables Humidity9am et Humidity3pm donnent les pourcentages d'humidité dans l'air mesurés toujours à 9h du matin et à 15h de l'après-midi. 
    Ces quatre variables seront conservées, toutes les quatre, car nous ne pouvons affirmer que Pressure3pm soit redondante avec Pressure9am, et vice-versa.\n 
    
    De même pour les deux variables Humidity9am et Humidity3pm.
    
    En raison de la dépendance des variables Cloud9am avec Humidity9am et Cloud3pm avec Humidity3pm, et surtout du très grand nombre de valeurs manquantes dans les variables Cloud9am et Cloud3pm, nous avons décidé d'abandonner Cloud9am et Cloud3pm.\n\n

    
    """)
    
    st.image("images//VariablesAband.png")

    st.markdown("""
    Nous nous séparons de Katherine et Launceston. Il y a trop peu de données exploitables et pas de ville radar pertinentes pour ces 2 villes.\n
    Par ailleurs nous nous séparons aussi des années incomplètes (qui ne s’étalent pas du 1er janvier au 31 décembre).\n
    Enfin nous avons supprimé les lignes où RainToday et/ou RainTommorrow comportent des valeurs manquantes NaNs.\n\n


    """)

    st.image("images//variable3.png")
    st.markdown("")
    st.markdown("")





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
    
    st.markdown("")
    st.markdown("")



    st.header("Traitement des valeurs manquantes")

    st.markdown("""

    Plusieurs méthodes s’offraient à nous. Remplir avec la méthode bfill(), ffill(), le mode() ou le mean(). Nous avons opté pour une méthode plus réaliste et travaillé sur la base de la proximité des villes et le fait que le temps soit similaire. 



    Nous avons parcouru les sites donnés dans la fiche projet et nous avons principalement exploité le site météo du gouvernement australien  (http://www.bom.gov.au) d’où sont tirées les données météorologiques du pays. 



    Le principe est de fonctionner sur la notion de radar : zone de quelques kilomètres autour d’une ville où l’estimation du temps s’applique aux villes environnantes. 

    Nous avons repris cette notion de ville radar pour notre projet et l’avons appliqué à notre jeu de données (ci-dessous exemple avec Moree).\n

    """)

    st.image("images//carteradar.png")

    st.markdown("")
    st.markdown("")
    
    st.image("images//carteaustralie1.png")
    
    
    st.markdown("")
    st.markdown("")
    st.markdown("""
    Ci-dessous la répartition des villes radar de notre dataset (en rouge)
     """)


    st.image("images//carteaustralie2.png")
    
    st.markdown("""
    Ci-dessous la répartition des villes radar de notre dataset (en rouge)
     """)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    st.header("Etude des séries temporelles")
    
    st.markdown("""
    Le résultat n’est pas concluant, le manque de mémoire fait planter le traitement. Réduction du nombre de lignes à traiter en enlevant quelques années (on a conservé 3 ans et demi), mais le résultat reste le même bien que la mémoire cherchant à être allouée est de 4Go.  Passer en dessous de 3 ans et demi est moins pertinent pour ce modèle, nous avons donc abandonné cette piste.\n

    """)
    
    st.image("images//encartST.png")
    
    
    
if page == pages[3]: 

    st.header("Algorithme choisi")

    st.markdown("""

    Nous avons donc créé plusieurs fonctions pour le traitement des données manquantes.

    On travaille par variable et on traite les données manquantes de chaque ville sur ce principe de ville radar.



    Ce traitement est relativement long (quelques minutes) sur les variables où il y a des absences quasi totales de modalités (Sunshine ou Evaporation ou Pressure9am et Pressure3pm). Pour le reste les temps d’exécution sont bons. Algorithme de traitement en masse synthétisé ci-dessous	



    """)

    st.image("images//algorithme.png")



if page == pages[4]: 

    st.header("Entrainement des différents modèles")

    st.markdown("""

    Nous avons commencé par travailler sur l’ensemble des données du territoire australien, qui regroupe, une fois les données nettoyées, 133 963 lignes, en lançant une Logistic Regression naïve. Les résultats ont été encourageants, mais pas optimaux, nous avons donc décidé de poursuivre avec ce modèle en essayant de trouver les meilleurs paramètres, grâce à une instruction de GridSearch.\n



    Conscients que le jeu de données est déséquilibré (22% des données ont la variable « Rain Tomorrow », qui est la variable à prédire, à Yes - ou à 1 -), nous avons pensé à tester aussi le (ou les) modèle(s) de Machine Learning avec  un rééchantillonnage, grâce à OverSampler.\n



    Souhaitant trouver de meilleures performances pour les prédictions en Machine Learning, nous avons pensé tester d’autres modèles de classification et nous avons choisi tout d’abord le modèle des voisins les plus proches (les K Nearest Neighbors).\n



    Et dans la même logique que pour la Logistic Regression, nous avons commencé par entraîner le modèle (pour lequel nous avons recherché les « Best Params », grâce à la fonction GridSearchCV), sur les données échantillonnées de façon classique en réservant 20% pour le jeu de test.  Nous avons évalué ses performances. Puis nous l’avons ré-entrainé sur un échantillon oversamplé pour la classe minoritaire, qui est la classe positive, et à nouveau nous avons mesuré les performances.\n



    Vu que les résultats du KNN étaient meilleurs que ceux de la Logistic Regression, cela nous a encouragé à tester un troisième classifieur, pour un troisième modèle.\n



    Et nous avons pensé au Random Forest Classifier, issu des arbres de décision.\n



    Evidemment, nous aurions pu déterminer les « Best estimators » avec les « Best Parameters », mais avec GridSearchCV, les temps de traitement étaient beaucoup trop longs, et rien que pour un seul classifieur, trouver les meilleurs paramètres pour l’ensemble du jeu de données de l’Australie (133963 lignes), le PC mettait plus d’une journée pour donner des résultats. Pour une raison de temps de calcul et de capacité de nos machines, nous avons renoncé à sélectionner les meilleurs classifieurs de cette façon.\n



    Et nous avons opté de tester donc Random Forest, avec les meilleurs paramètres. Tout comme dans la logique de la Logistic Regression et du KNN, nous avons l’avons entrainé sur deux jeux de données différents, l’un sélectionné par la fonction « train_test_split », et l’autre jeu d’entraînement ayant subi un suréchantillonnage avec OverSampler.\n

    \n



    """)
    st.markdown("")
    st.markdown("")

    st.subheader("Résultats des entrainements")

    st.markdown("""

    Pour tout le dataset avec et sans OverSampling, nous avons obtenus les résultats suivants:

    """)

    st.image("images//modele1.png")



    st.markdown("""

    Ne voulant pas tirer de conclusions immédiates sur ces résultats, où nous aurions pu dire que le KNN SANS Oversampling nous paraissait être le meilleur estimateur, ayant les meilleures performances pour la variable cible minoritaire (positive , à 1), mais aussi le Meilleur taux de performance moyen pour toutes les predictions positives et negatives. Nous avons souhaité aller au-delà, et approfondir nos tests.\n



    Pour cela, nous avons pensé diviser notre jeu de données, par régions, ces régions qui ne sont autres que les regions administratives de l’Australie, et qui regroupent, chacune au moins un centre météorologique, appelé aussi “radar”, cela permettait d’alléger le jeu de données, mais aussi de pouvoir distinguer s’il y a des spécificités liées aux regions ou aux villes.\n





    Et nous avons décidé de re-tester les 3 précédents classifieurs déjà décrits plus haut, sur de plus petits jeux de données, liés directement aux régions suivantes:\n



    SA : South Australia\n

    WA: Western Australia\n

    NSW: New South Wales\n

    VIC: Victoria\n

    TAS/NIS: Tasmania / Norfolk Islands\n

    Les régions du Queensland et du North, situées au Nord du continent ont été retirées des calculs, car la quantité des données figurant dans le dataset d’origine n’était pas suffisante.



    """)
    st.markdown("")
    st.markdown("")


    st.subheader("South Australia et Western Australia")

    st.markdown("""
    L’entrainement de données a été fait avec et sans over-sampling. Les 2 dataset ont été entrainés séparemment. Enfin, Nous avons lancé un GridCVSearch afin de trouver les meilleurs paramètres.\n
    
    Le dataset South Australia comportait 8853 lignes et celui de Western Australia 23 203 lignes.\n
    
    La Logistic Regression  avec Oversampling donne pratiquement les mêmes résultats que Random Forest avec Oversampling et dans ce cas, vaut mieux adopter le modèle ayant le traitement le plus rapide.\n\n

    """)
    
    st.image("images//modele2.png")
    st.markdown("")
    st.markdown("")
    
    
    
    st.subheader("Tasmanie et Norfolk Island")

    st.markdown("""

    L’entrainement de données a été fait avec et sans over-sampling. Les 2 dataset ont été réunis car les données étaient peu nombreuses et le dataset final comporte 5905 lignes. Les modèles ont été entrainés sur les 5905 lignes. Enfin, Nous avons lancé un GridCVSearch afin de trouver les meilleurs paramètres.

    """)

    st.image("images//modele3.png")
    st.markdown("")
    st.markdown("")



    st.subheader("New South Wales et Victoria")

    st.markdown("""

    Comme mentionné dans la section précédente, les données de la variable catégorielle, RainTomorrow, ne sont pas équilibrées. Par conséquent, cela peut entraîner des prédictions peu fiables sur les données examinées. Comme indiqué précédemment dans la section sur les modèles de classification des îles, il a été décidé de traiter les données par la méthode du suréchantillonnage. En outre, compte tenu des prévisions au niveau national, la méthode de régression logistique était coûteuse en termes de calcul. En combinant ces considérations, il a été décidé que pour les régions administratives de Victoria (VIC) et de New South Wales (NSW), les méthodes de classification à employer sont KNN et RandomForest. En outre, notre décision est motivée par le fait que ces deux régions ont le plus grand nombre de stations météorologiques, ce qui pourrait les rendre représentatives de l'ensemble de la base de référence. Afin d'obtenir une image claire, nous avons d'abord choisi de traiter ces deux régions administratives séparément, puis de les traiter ensemble. 

    Le dataset New South Wales comportait xx lignes et celui de Victoria xx lignes
    """)

    st.image("images//modele4.png")
    st.markdown("")
    st.markdown("")



    st.subheader("Synthèse des entrainements")
    
    st.markdown("""
    \n\n
    """)

    st.image("images//synthese.png")



if page ==pages[6]:

    st.header("Restitution graphique des prédictions")

    st.markdown("""

    Nous avons choisi Folium pour la simplicité de son usage et le rendu graphique.\n

    Nous avons donc récupéré la longitude et latitude de chaque ville et l’avons stockée dans une table.\n

    Associée à la notion de radar, nous représenterons les villes radar d’une couleur différente (en rouge).\n

    Nous avons donc construit un tableau unique qui prend en compte les villes, les radars et la variable RainTomorrow. Bien que la variable radar n'apparaisse pas dans la carte géographique, il a évolué pour l'inclure afin de faire un double contrôle sur la manipulation correcte du dataframe. La base de données résultante a ensuite été filtrée afin de comparer les données stockées dans la trame de données source avec les données prédites à la même date. \n

    Si le temps est beau alors nous figurerons un soleil, sinon une goutte d’eau.\n

    Ci-dessous, une image de la variable RainTomorrow pour la journée du 2017/04/2011.



    """)

    st.image("images//graphique1.png")
    st.markdown("")
    st.markdown("")

    st.markdown("""

    Le jeu de base utilisé est le jeu complété, c'est-à-dire celui dans lequel certaines villes ont été retirées parce que la variable RainTomorrow n'a pas été enregistrée expérimentalement à cette date spécifique, il n'est pas surprenant que certaines villes ne soient pas présentes sur la carte.\n

    Comme cet exemple de démonstration est réalisé sur la région de New South Wales, la prévision pour les villes situées dans cette région est présentée ci-dessous.

    """)

    st.image("images//graphique2.png")
    st.markdown("")
    st.markdown("")



    st.markdown("""

    On observe que le 2017/04/11, il a plu sur le littoral de la région de New South Wales, plus précisément autour de la ville de Sidney. En revanche, dans les villes les plus proches (par exemple), la journée a été ensoleillée. Cela crée une zone limite très fine et la précision de la méthode en question est donc mise à l'épreuve.  



    Nous tenons à souligner que dans notre méthode d'apprentissage automatique, la date n'était pas indiquée dans l'échantillon X_test. Pour cette raison, une fois que le modèle a été entraîné et que la variable a été prédite, un cadre de données contenant les prédictions a été construit et ensuite fusionné avec le cadre de données X_test par la méthode de fusion afin d'avoir un cadre de données complet contenant les emplacements. Les variables latitude, longitude et toutes les autres variables présentes dans le cadre de données original, filtrées par date, ont ensuite été ajoutées à ce cadre de données.



    Voici le résultat de la prédiction sur la région NSW.



    """)

    st.image("images//graphique3.png")
    st.markdown("")
    st.markdown("")

    st.markdown("""

    Comme on peut le voir sur la figure, la variable RainTomorrow le 2017/04/11 n'a été prédite que pour 3 villes (Albury, Canberra, et Richmond). Ceci est cohérent avec le fait que lors de l'étape de division de l'ensemble de données en test et formation, l'échantillon de formation représente 20% de l'ensemble de données.

    """)



if page == pages[5]:

    st.header("Conclusions")

    st.markdown("""

    Ce projet de master s'est concentré sur la prédiction de la variable rainTomorrow, et a permis de mettre en pratique les concepts étudiés dans le cadre de la formation Data Scientest. Afin de prédire la variable, des protocoles ont été développés sur la base de l'état de l'art en matière de calcul. Afin de développer ces protocoles, un traitement du jeu de données a d'abord été effectué, suivi d'une analyse statistique et graphique, permettant d'obtenir une vue d'ensemble.\n\n

    Comme les prévisions météorologiques peuvent suivre des cycles dans le temps, une étude des séries temporelles a été entreprise. Cependant, il faut savoir que le manque de mémoire de nos ordinateurs fait planter le traitement. Cela peut être imputable à la taille de l'ensemble de données ; ce résultat a également été obtenu dans le cas d'un ensemble de données réduit (3 à 5 ans). Les séries temporelles nous auraient permis de prévoir à la fois RainTomorrow mais aussi MaxTemp et MinTemp. 

    """)

    st.image("images//conclusion1.png")
    st.markdown("")

    st.markdown("""

    Nous nous sommes donc contentés d’évaluer les performances des prédictions des classifieurs que nous avons testés. Les meilleurs hyperparamètres varient parfois selon les régions administratives. En revanche, les hyperparamètres de la méthode de la forêt aléatoire sont les mêmes pour toutes les régions. De plus, cette méthode s'est avérée être la plus rapide (en termes de temps réel) par rapport aux deux autres méthodes.

    """)

    st.image("images//conclusion2.png")
    st.markdown("")

    st.markdown("""

    En conclusion, notre étude nous a permis à la fois de déterminer la variable catégorielle en question avec une bonne précision et de déterminer les hyperparamètres optimaux. \n\n

    Enfin, il faut souligner que l'utilisation de la bibliothèque missingo montre que l'ensemble de la base de données est caractérisé par un nombre élevé de valeurs manquantes.\n\n


    """)
    
    st.image("images//conclufin.png")
    st.markdown("")

if page == pages[7]:
    st.header("Entrainez nos modèles !")

    region  = ["New South Wales", "Western Australia", "Autralia"]
    model  = ["KNN", "Random Forest", "Logistic Regression"]
    # overSampl = ["Avec", "Sans"]

    selectionRegion = st.selectbox('Choisissez une région', options = region)
    selectionModel = st.selectbox('Choisissez un modèle', options = model)
    # over = st.radio("Votre choix OverSampling", options= overSampl)
    over = st.radio("Votre choix OverSampling", ('Avec', 'Sans'))
    
    st.markdown("""
    Résultats de l'entrainement
    \n\n
    """)
    st.markdown("")
    st.markdown("")

    if selectionRegion == region[0] and selectionModel == model[0] and over == 'Avec':
        st.image("images//nsw_knn_avec_over.png")
    elif selectionRegion == region[0] and  selectionModel == model[0] and over == 'Sans':
        st.image("images//nsw_knn_sans_over.png")
    elif selectionRegion == region[0] and  selectionModel == model[1] and over == 'Avec':
        st.image("images//nsw_random_avec_over.png")
    elif selectionRegion == region[0] and  selectionModel == model[1] and over == 'Sans':
        st.image("images//nsw_random_sans_over.png")
    elif selectionRegion == region[1] and selectionModel == model[0] and over == 'Sans':
        st.image("images//wes_knn_sans_over.png")
    elif selectionRegion == region[1] and selectionModel == model[0] and over == 'Avec':
        st.image("images//wes_knn_avec_over.png")
    elif selectionRegion == region[1] and selectionModel == model[1] and over == 'Avec':
        st.image("images//wes_rand_avec_over.png")
    elif selectionRegion == region[1] and selectionModel == model[1] and over == 'Sans':
        st.image("images//wes_rand_sans_over.png")
    elif selectionRegion == region[2] and selectionModel == model[0] and over == 'Avec':
        st.image("images//knn_avec_over.png")
    elif selectionRegion == region[2] and selectionModel == model[0] and over == 'Sans':
        st.image("images//knn_sans_over.png")
    elif selectionRegion == region[2] and selectionModel == model[1] and over == 'Avec':
        st.image("images//random_forest_avec_over.png")
    elif selectionRegion == region[2] and selectionModel == model[1] and over == 'Sans':
        st.image("images//random_forest_sans_over.png")
    elif selectionRegion == region[2] and selectionModel == model[2] and over == 'Avec':
        st.image("images//logistique_reg_avec_over.png")
    elif selectionRegion == region[2] and selectionModel == model[2] and over == 'Sans':
        st.image("images//logistique_reg_sans_over.png")


if page == pages[8]:
    
    st.header("Nous allons voir une prédiction pour le 11 avril 2017")
    
    region=st.selectbox("Choisissez une région", ('New South Wales', 'Western Australia'))
    
    st.markdown("")
    st.markdown("")
    
    
           
    st.write('Prédiction du temps sur la région ', region, ' avec RandomForest et OverSampling')
       
    villesref.rename(columns = {'Ville':'city'}, inplace = True)    
    a_clean_f = australian_clean[australian_clean['Date']=='2017-04-11']
    a_clean_f.replace('No', 0, inplace=True)
    a_clean_f.replace('Yes', 1, inplace=True)
    a_clean_f.rename(columns = {'Location':'city'}, inplace = True)
    df_cd = pd.merge(a_clean_f, villesref, how='inner')
    df_cd_1 = pd.merge(df_cd, df_villes)
    df_villes_ref={}
    for idx,lige in villesref.iterrows():
        df_villes_ref[lige['city']]= lige['Radar']

    url="images/{}".format
    urlimage=url("IconSun.png")
        
    if region=='New South Wales' :
        datacopy = australian_clean.copy()
        datacopy.index.name='number_row'
        datacopy.reset_index(inplace=True)
        result_NSW.rename(columns = {'RainTomorrow':'RainTomorrowPred'}, inplace = True)
        df_result = result_NSW[['number_row', 'RainTomorrowPred']].copy()
        datacopy.rename(columns = {'Location':'city'}, inplace = True)
        df_result_1 = datacopy.merge( right=df_result, on='number_row', how='inner')
        df_result_2 = df_result_1[df_result_1['Date']=='2017-04-11']
        df_result_3 = pd.merge(df_result_2, df_villes)
        df_result_4 = pd.merge(df_result_3, villesref)
            
        m=folium.Map(location=[df_cd_1[df_cd_1['city']=='Sydney']['lat'],df_cd_1[df_cd_1['city']=='Sydney']['lng']],zoom_start=5)

            

        for index, row in df_result_4.iterrows():
            if df_villes_ref[row['city']]=='isole':
                colradar='blue'
                pos=folium.features.CustomIcon(urlimage,icon_size=(30,30))
            else:
                colradar='blue'
                pos=folium.features.CustomIcon(urlimage,icon_size=(30,30))
            if row['RainTomorrowPred'] == 1:
                Libelle=" il pleuvra"
                popup="<h2>"+row['city']+"</h2><p>demain"+ Libelle+"</p>"
                marker=folium.Marker(location=[row['lat'], row['lng']],popup=popup,tooltip=row['city'],icon=folium.Icon(icon='tint',icon_color='blue',color=colradar))
                marker.add_to(m)
            else:
                Libelle=" il ne pleuvra pas"
                popup="<h2>"+row['city']+"</h2><p>demain"+ Libelle+"</p>"
                marker=folium.Marker(location=[row['lat'], row['lng']],popup=popup,tooltip=row['city'],icon=pos)
                marker.add_to(m)
        st_data = st_folium(m,width=700)

            
    if region=='Western Australia' :
        result_WEA.rename(columns = {'RainTomorrow':'RainTomorrowPred'}, inplace = True)
        df_result_5 = result_WEA[['city', 'RainTomorrowPred']].copy()
        df_result_6 = pd.merge(df_result_5, df_villes)
        df_result_7 = pd.merge(df_result_6, villesref)
        m=folium.Map(location=[df_cd_1[df_cd_1['city']=='PerthAirport']['lat'],df_cd_1[df_cd_1['city']=='PerthAirport']['lng']],zoom_start=5)


        for index, row in df_result_7.iterrows():
    
            if df_villes_ref[row['city']]=='isole':
                colradar='blue'
                pos=folium.features.CustomIcon(urlimage,icon_size=(30,30))
            else:
                colradar='blue'
                pos=folium.features.CustomIcon(urlimage,icon_size=(30,30))
            if row['RainTomorrowPred'] == 1:
                Libelle=" il pleuvra"
                popup="<h2>"+row['city']+"</h2><p>demain"+ Libelle+"</p>"
                marker=folium.Marker(location=[row['lat'], row['lng']],popup=popup,tooltip=row['city'],icon=folium.Icon(icon='tint',icon_color='blue',color=colradar))
                marker.add_to(m)
            else:
                Libelle=" il ne pleuvra pas"
                popup="<h2>"+row['city']+"</h2><p>demain"+ Libelle+"</p>"
                marker=folium.Marker(location=[row['lat'], row['lng']],popup=popup,tooltip=row['city'],icon=pos)
                marker.add_to(m)
            
        st_data = st_folium(m,width=700)
