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



pages = ["Introduction", "Datasets", "Exploration des donn??es et visualisation", "Algorithme de traitement des NaN", "Mod??lisation", "Conclusions","Restitution graphique des pr??dictions", "Entrainez nos mod??les ! ","La pr??diction du 11 avril 2017 !"]



page = st.sidebar.radio("Menu", options = pages)



if page == pages[0]:

    st.image("images//image001.jpg")
    

    st.markdown("""


Projet pr??sent?? par Zeina ACHKAR, Sandrine ASSERAF, Bernardino TIRRI et Magarh TSILOUONI\n
Promotion Mai 2022 DataScientest mode Bootcamp\n\n
     
       
       
 """)
    st.markdown("")
    st.markdown("")
    st.markdown("""
\n\nLe dataset contient environ 10 ans d'observations m??t??orologiques quotidiennes provenant de nombreux endroits en Australie.

Il y a donc diff??rentes visualisations int??ressantes possibles.

Le premier objectif serait de pr??dire la variable cible : RainTomorrow. Elle signifie : a-t-il plu le jour suivant, oui ou non ? Cette colonne est Oui si la pluie pour ce jour ??tait de 1mm ou plus.

Dans un second temps, on pourra effectuer des pr??dictions ?? long terme, en utilisant des techniques math??matiques d???analyse de s??ries temporelles.



Data 

-	https://www.kaggle.com/jsphyg/weather-dataset-rattle-package



Benchmark/ Bibliographie/Source:

-	Les observations proviennent de nombreuses stations m??t??orologiques. Les observations quotidiennes sont disponibles sur http://www.bom.gov.au/climate/data.

-	Un exemple des derni??res observations m??t??orologiques ?? Canberra : http://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml

-	D??finitions adapt??es de http://www.bom.gov.au/climate/dwo/IDCJDW0000.shtml

-	Source des donn??es : http://www.bom.gov.au/climate/dwo/ et http://www.bom.gov.au/climate/data.

    """)



if page == pages[1]:

    st.markdown("""

    Nous avons travaill?? sur un fichier csv (weatherAUS.csv). Ce dataset fournit des donn??es m??t??orologiques de 49 villes d???Australie entre 2009 et 2017. Le jeu de donn??es initial comportait 145 460 lignes et 23 colonnes. 

    La variable ?? pr??dire est RainTomorrow. Elle est renseign??e ?? Yes ou No.



Les ann??es 2007 2008 et 2017 sont tr??s incompl??tes. Les autres ann??es sont compl??tes bien qu???il puisse manquer quelques jours dans certains mois et certaines villes.

Le manque de donn??es se concentre beaucoup sur Sunshine et Evaporation ou les variables Pressure ce qui nous a pos?? probl??me pour identifier une m??thode fiable de comblement des NaN.



Certaines variables marchent par couple :  donn??es ?? 9 heures du matin et une autre ?? 3 heures de l???apr??s-midi. C???est le cas de :

-   Cloud

-   Pressure

-   Windir

-   WinSpeed

-   Temp



    """)

    st.dataframe(old_data)

    st.markdown("")
    st.markdown("")

    resumer = st.checkbox('Afficher le r??sum??')

   



    if resumer:

        st.dataframe(old_data.describe())



    st.markdown("""

        Les villes ne sont pas ??quitablement r??parties sur le territoire. Le centre du pays est clairement sous repr??sent??.

        """)

    st.image("images//carte_villes.png")



if page == pages[2]:

    st.header("Analyse des donn??es")

    st.markdown("""

    Pour mener ?? bien un projet de data science, nous proc??dons toujours ?? une premi??re ??tape qui est l'analyse des donn??es. Cette ??tape constitue un enjeu majeur pour une analyse approfondie des donn??es brutes pour rechercher des mod??les, des tendances et des mesures dans un ensemble de donn??es.

    

    Description des variables du dataset  

    """)

    st.image("images//variable1.png")

    st.image("images//variable2.png")
    st.markdown("")
    st.markdown("")



    st.header("Analyse statistique du dataset")

    st.markdown("""

    Apr??s la phase de pr??paration des donn??es, une premi??re analyse statistique de la base de donn??es a ??t?? r??alis??e en utilisant la m??thode describe. Le r??sultat est pr??sent?? ci-dessous :    """)

    st.write(old_data.describe())

    

    st.markdown("""

    Sur une p??riode de 8 ans, le tableau montre que les valeurs moyennes des temp??ratures minimales et maximales sont respectivement de 12 et 23 degr??s Celsius, les valeurs moyennes de pression enregistr??es le matin et l'apr??s-midi sont de 1017 et 1015 et la quantit?? moyenne de pluie est de 2,34 mm. Bien que la plupart des stations m??t??orologiques soient situ??es sur la c??te, o?? les pr??cipitations sont plus fr??quentes que dans la zone centrale, l'Australie est un pays principalement aride et, dans l'ensemble, le pays n'est pas caract??ris?? par de fortes pr??cipitations. Les quartiles sont ??galement indiqu??s dans le tableau. Une fa??on plus simple d'analyser les outils est de repr??senter graphiquement les donn??es de la trame de donn??es dans ce que l'on appelle le graphique ?? moustaches. Le graphique est pr??sent?? ci-dessous.""")   



    st.image("images/boxplot.png")

    st.markdown("""

    En analysant le graphique, on constate que pour les variables Pluie et ??vaporation, les valeurs extr??mes sont situ??es au-dessus du troisi??me quartile. Conform??ment au tableau ci-dessus, nous pouvons d??duire que ce n'est qu'?? certaines p??riodes de l'ann??e que la quantit?? de pr??cipitations est sup??rieure ?? la moyenne. Il en va de m??me pour la variable Evaporation et la variable Vitesse du vent. Contrairement ?? ces variables, la variable humidit?? enregistr??e ?? 9h du matin est caract??ris??e par un faible nombre d'outils, qui se situent en dessous du premier quartile. En entrant dans le d??tail, il est int??ressant de noter o?? les plus fortes pr??cipitations ont ??t?? enregistr??es en huit ans d'??chantillonnage. Le graphique cumulatif de la variable "pr??cipitations" pour les diff??rentes stations m??t??orologiques et ann??es est pr??sent?? ci-dessous.

    """)

    st.image("images//region.png")

    st.markdown("""

    Les graphiques montrent que les villes caract??ris??es par de fortes pr??cipitations sont Cairns, Darwin, Coffs Harbour, Gold Coast et enfin Wollongong. Ces stations m??t??orologiques sont situ??es respectivement dans les r??gions NSW, QUE, VIC, WAU et NTE. Si l'on prend la variable RainFall comme indicateur, les ann??es caract??ris??es par de fortes pr??cipitations sont 2010 et 2011. Les graphiques cumulatifs de la variable cat??gorielle RainToday et RainTomorrow sont pr??sent??s ci-dessous.

    """)

    st.image("images//region2.png")



    st.markdown("""

    Le graphique ci-dessous montre une relation lin??aire ??troite entre les deux variables consid??r??es.

    """)

    sns.pairplot(data=dfW, vars=('Pressure3pm','Pressure9am'), hue='RainTomorrow' )

    st.pyplot(plt.show())



    st.markdown("""

    Le graphique ci-apr??s montre la temp??rature maximale en d??gr?? celcius par rapport ?? la temp??rature minimale enregistr??e en 24 heures jusqu'?? 9 heures du matin en d??gr?? celcius.

    """)

    sns.pairplot(data=dfW, vars=('MinTemp','MaxTemp'), hue='RainTomorrow' )

    st.pyplot(plt.show())

    st.markdown("""

    Le graphique ci-dessus montre une relation lin??aire entre les variables prises en consid??ration. La temp??rature minimale est d'environ -8 d??gr?? celcius et la maximale est d'environ 48 d??gr?? celcius. La carte satellite de l'Australie montre que, dans les r??gions du sud-est et du sud-ouest, le climat est plus temp??r??, rendant l'air propice ?? l'implantation humaine ; c'est dans ces r??gions que l'on trouve les grandes villes australiennes telles que Sydney, Perth ou Melbourne. ??tant donn?? que l'Australie est situ??e dans l'h??misph??re sud, la diff??rence de temp??rature d'une r??gion ?? l'autre est moins prononc??e, mais en raison de son immensit?? et de son climat essentiellement aride, les ??carts de temp??rature peuvent ??tre importants.

    """)
    
    st.markdown("")
    st.markdown("")



    st.header("Tests de d??pendances des variables")

    st.subheader("CHI2 et V de Cramer")

    st.markdown("""

    Nous avons fait de nombreux tests afin d?????valuer les d??pendances entre variables. Le code suivant a ??t?? ex??cut?? pour 5 variables en lien avec RainTomorrow. Nous ne mettrons pas le d??tail d???ex??cution de chaque variable, mais juste les r??sultats.

    """)

    st.markdown("""

    df['WindGustDir'], df['RainTomorrow'] pas de forte corr??lation\n

    df['WindGustSpeed'], df['RainTomorrow'] pas de forte correlation\n

    df['Humidity3pm'], df['RainTomorrow'] corr??lation Moyenne\n

    df['Sunshine'], df['RainTomorrow'] corr??lation Moyenne\n

    df['RainToday'], df['RainTomorrow'] corr??lation Moyenne\n

    """)

    st.subheader("Tests ANOVA")

    st.markdown("""

    Pour les variables comprenant de nombreuses modalit??s nous avons effectu?? des tests Anova :

    MinTemp, MaxTemp Evaporation et Pressure3pm



    Si la p_value (PR>F) est <5%, on rejette l'hypoth??se H0 qui dit que les 2 variables X et Y sont ind??pendantes et on d??duit du test que X a un effet statistique significatif sur la variable cible.



    Pressure 3pm a un effet significatif sur RainTomorrow

    MinTemp a un effet statistique significatif sur RainTomorrow

    MaxTemp a un effet statistique significatif sur RainTomorrow

    Evaporation a un effet statistique significatif sur RainTomorrow



    Nous avons aussi fait les tests HI2 et Anova entre Cloud et Humidity, pour jauger la corr??lation entre ces variables. Les tests ont montr?? un effet statistique significatif entre Cloud9am et Humidity9am et entre Cloud3pm et Humidity3pm.\n
    Pour ces raisons de "non ind??pendance" des variables Cloud9am et Humidity9am,  puis de Cloud3pm et Humidity3pm, et sachant que Cloud9am et Cloud3pm comportent pr??s de 60 000 valeurs de NaN chacune (au niveau du dataset total qui comportent pr??s de 142 000 lignes), nous avons d??cid?? de nous s??parer de ces deux variables Cloud9am et Cloud3pm.\n\n



    """)
    st.markdown("")
    st.markdown("")



    st.header("S??lection des meilleures features")

    st.markdown("""

    Nous avons lanc?? un SlectFromModel afin d???identifier les meilleures features pour nos pr??dictions sur le jeu de donn??es Tasmanie et Norfolk Island. Voici le r??sultat :

    """)

    st.image("images//SelectBest.png")

    st.markdown("""

    

    Nous avons choisi de conserver toutes les variables car il y en a assez peu dans le dataset, mais nous aurons au moins fait cet exercice qui confirme assez les tests de Cramer fait ??galement.



    """)
    st.markdown("")
    st.markdown("")
    



    st.header("Selection des variables et des donn??es ?? abandonner")

    st.markdown("""

    MaxTemp est la temp??rature maximale de la journ??e. MinTemp est la temp??rature minimale de la journ??e. 

    Temp9am est la temp??rature mesur??e ?? 9 heures du matin. Temp3pm est la temp??rature mesur??e ?? 15 h.\n 
    
    L'information des temp??ratures ?? 9h du matin et ?? 15h de l'apr??s-midi semblent redondantes avec les infos des temp??ratures extr??mes de la journ??e (qui semblent plus importantes). 
    Nous avons choisi de garder MaxTemp et MinTemp et d'abandonner Temp9am et Temp3pm.\n

    Nous constatons la m??me chose pour la Direction du Vent (WindGustDir, WindDir9am, WindDir3pm), et la Vitesse du Vent (WindGustSpeed, WindSpeed9am, WindSpeed3pm), o?? nous avons les variables WindGustDir  d??signant la direction du vent le plus fort de la journ??e, et WindGustSpeed la vitesse du vent le plus fort de la journ??e, qui semblent donner des infos plus importantes, que les variables regroupant les observations du vent ?? 9h du matin et 15 de l'apr??s-midi. 
    Pour cette raison, nous avons choisi de garder WindGustDir et WindGustSpeed et d'abandonner WindDir9am, WindDir3pm,  WindSpeed9am, WindSpeed3pm.\n
    
    Les variables Pressure9am et  Pressure3pm donnent les niveaux de la Pression Atmosph??rique ?? 9h du matin et ?? 15h de l'apr??s-midi.\n 
    
    Les variables Humidity9am et Humidity3pm donnent les pourcentages d'humidit?? dans l'air mesur??s toujours ?? 9h du matin et ?? 15h de l'apr??s-midi. 
    Ces quatre variables seront conserv??es, toutes les quatre, car nous ne pouvons affirmer que Pressure3pm soit redondante avec Pressure9am, et vice-versa.\n 
    
    De m??me pour les deux variables Humidity9am et Humidity3pm.
    
    En raison de la d??pendance des variables Cloud9am avec Humidity9am et Cloud3pm avec Humidity3pm, et surtout du tr??s grand nombre de valeurs manquantes dans les variables Cloud9am et Cloud3pm, nous avons d??cid?? d'abandonner Cloud9am et Cloud3pm.\n\n

    
    """)
    
    st.image("images//VariablesAband.png")

    st.markdown("""
    Nous nous s??parons de Katherine et Launceston. Il y a trop peu de donn??es exploitables et pas de ville radar pertinentes pour ces 2 villes.\n
    Par ailleurs nous nous s??parons aussi des ann??es incompl??tes (qui ne s?????talent pas du 1er janvier au 31 d??cembre).\n
    Enfin nous avons supprim?? les lignes o?? RainToday et/ou RainTommorrow comportent des valeurs manquantes NaNs.\n\n


    """)

    st.image("images//variable3.png")
    st.markdown("")
    st.markdown("")





    st.header("Ajout des nouvelles variables")

    st.markdown("""

    Comme nous devrons travailler ?? la journ??e, nous avons d??cid?? de d??couper la date en 3 variables suppl??mentaires : le jour, le mois et l???ann??e.

    """)



    st.header("Identification et r??partition des NaNs")

    st.markdown("""

    Nous avons travaill?? avec le module MissingNo et nous avons analys?? ville par ville qu???elles ??taient les valeurs absentes et dans quelles proportions. 

    Les valeurs manquantes se concentrent beaucoup sur 2 couples de valeurs : Evaporation et Sunshine et Pressure9am et Pressure3pm. Il y a un lien fort entre Evaporation et Sunshine quand il s???agit de valeurs manquantes. La m??me chose se constate pour les deux variables Pressure. Certaines villes n???ont aucune valeur pour le couple de variables !\n

    Vous trouverez ci-apr??s quelques figures pour vous donner une id??e de la r??partition des donn??es manquantes par ville.

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
    st.markdown("")



    st.header("Traitement des valeurs manquantes")

    st.markdown("""

    Plusieurs m??thodes s???offraient ?? nous. Remplir avec la m??thode bfill(), ffill(), le mode() ou le mean(). Nous avons opt?? pour une m??thode plus r??aliste et travaill?? sur la base de la proximit?? des villes et le fait que le temps soit similaire. 



    Nous avons parcouru les sites donn??s dans la fiche projet et nous avons principalement exploit?? le site m??t??o du gouvernement australien  (http://www.bom.gov.au) d???o?? sont tir??es les donn??es m??t??orologiques du pays. 



    Le principe est de fonctionner sur la notion de radar : zone de quelques kilom??tres autour d???une ville o?? l???estimation du temps s???applique aux villes environnantes. 

    Nous avons repris cette notion de ville radar pour notre projet et l???avons appliqu?? ?? notre jeu de donn??es (ci-dessous exemple avec Moree).\n

    """)

    st.image("images//carteradar.png")

    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    st.image("images//carteaustralie1.png")
    st.markdown("""
    Radars m??t??o de l'Australie
     """)
    
    
    st.markdown("")
    st.markdown("")
    st.markdown("")
    
        
    st.image("images//carteaustralie2.png")
    
    st.markdown("""
    Ci-dessus la r??partition des villes radar de notre dataset (en rouge)
     """)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    
    st.header("Etude des s??ries temporelles")
    
    st.markdown("""
    Le r??sultat n???est pas concluant, le manque de m??moire fait planter le traitement. R??duction du nombre de lignes ?? traiter en enlevant quelques ann??es (on a conserv?? 3 ans et demi), mais le r??sultat reste le m??me bien que la m??moire cherchant ?? ??tre allou??e est de 4Go.  Passer en dessous de 3 ans et demi est moins pertinent pour ce mod??le, nous avons donc abandonn?? cette piste.\n

    """)
    
    st.image("images//encartST.png")
    
    
    
if page == pages[3]: 

    st.header("Algorithme choisi")

    st.markdown("""

    Nous avons donc cr???? plusieurs fonctions pour le traitement des donn??es manquantes.

    On travaille par variable et on traite les donn??es manquantes de chaque ville sur ce principe de ville radar.



    Ce traitement est relativement long (quelques minutes) sur les variables o?? il y a des absences quasi totales de modalit??s (Sunshine ou Evaporation ou Pressure9am et Pressure3pm). Pour le reste les temps d???ex??cution sont bons. Algorithme de traitement en masse synth??tis?? ci-dessous	



    """)

    st.image("images//algorithme.png")



if page == pages[4]: 

    st.header("Entrainement des diff??rents mod??les")

    st.markdown("""

    Nous avons commenc?? par travailler sur l???ensemble des donn??es du territoire australien, qui regroupe, une fois les donn??es nettoy??es, 133 963 lignes, en lan??ant une Logistic Regression na??ve. Les r??sultats ont ??t?? encourageants, mais pas optimaux, nous avons donc d??cid?? de poursuivre avec ce mod??le en essayant de trouver les meilleurs param??tres, gr??ce ?? une instruction de GridSearch.\n



    Conscients que le jeu de donn??es est d??s??quilibr?? (22% des donn??es ont la variable ?? Rain Tomorrow ??, qui est la variable ?? pr??dire, ?? Yes - ou ?? 1 -), nous avons pens?? ?? tester aussi le (ou les) mod??le(s) de Machine Learning avec  un r????chantillonnage, gr??ce ?? OverSampler.\n



    Souhaitant trouver de meilleures performances pour les pr??dictions en Machine Learning, nous avons pens?? tester d???autres mod??les de classification et nous avons choisi tout d???abord le mod??le des voisins les plus proches (les K Nearest Neighbors).\n



    Et dans la m??me logique que pour la Logistic Regression, nous avons commenc?? par entra??ner le mod??le (pour lequel nous avons recherch?? les ?? Best Params ??, gr??ce ?? la fonction GridSearchCV), sur les donn??es ??chantillonn??es de fa??on classique en r??servant 20% pour le jeu de test.  Nous avons ??valu?? ses performances. Puis nous l???avons r??-entrain?? sur un ??chantillon oversampl?? pour la classe minoritaire, qui est la classe positive, et ?? nouveau nous avons mesur?? les performances.\n



    Vu que les r??sultats du KNN ??taient meilleurs que ceux de la Logistic Regression, cela nous a encourag?? ?? tester un troisi??me classifieur, pour un troisi??me mod??le.\n



    Et nous avons pens?? au Random Forest Classifier, issu des arbres de d??cision.\n



    Evidemment, nous aurions pu d??terminer les ?? Best estimators ?? avec les ?? Best Parameters ??, mais avec GridSearchCV, les temps de traitement ??taient beaucoup trop longs, et rien que pour un seul classifieur, trouver les meilleurs param??tres pour l???ensemble du jeu de donn??es de l???Australie (133963 lignes), le PC mettait plus d???une journ??e pour donner des r??sultats. Pour une raison de temps de calcul et de capacit?? de nos machines, nous avons renonc?? ?? s??lectionner les meilleurs classifieurs de cette fa??on.\n



    Et nous avons opt?? de tester donc Random Forest, avec les meilleurs param??tres. Tout comme dans la logique de la Logistic Regression et du KNN, nous avons l???avons entrain?? sur deux jeux de donn??es diff??rents, l???un s??lectionn?? par la fonction ?? train_test_split ??, et l???autre jeu d???entra??nement ayant subi un sur??chantillonnage avec OverSampler.\n

    \n



    """)
    st.markdown("")
    st.markdown("")

    st.subheader("R??sultats des entrainements")

    st.markdown("""

    Pour tout le dataset avec et sans OverSampling, nous avons obtenus les r??sultats suivants:

    """)

    st.image("images//modele1.png")



    st.markdown("""

    Ne voulant pas tirer de conclusions imm??diates sur ces r??sultats, o?? nous aurions pu dire que le KNN SANS Oversampling nous paraissait ??tre le meilleur estimateur, ayant les meilleures performances pour la variable cible minoritaire (positive , ?? 1), mais aussi le Meilleur taux de performance moyen pour toutes les predictions positives et negatives. Nous avons souhait?? aller au-del??, et approfondir nos tests.\n



    Pour cela, nous avons pens?? diviser notre jeu de donn??es, par r??gions, ces r??gions qui ne sont autres que les regions administratives de l???Australie, et qui regroupent, chacune au moins un centre m??t??orologique, appel?? aussi ???radar???, cela permettait d???all??ger le jeu de donn??es, mais aussi de pouvoir distinguer s???il y a des sp??cificit??s li??es aux regions ou aux villes.\n





    Et nous avons d??cid?? de re-tester les 3 pr??c??dents classifieurs d??j?? d??crits plus haut, sur de plus petits jeux de donn??es, li??s directement aux r??gions suivantes:\n



    SA : South Australia\n

    WA: Western Australia\n

    NSW: New South Wales\n

    VIC: Victoria\n

    TAS/NIS: Tasmania / Norfolk Islands\n

    Les r??gions du Queensland et du North, situ??es au Nord du continent ont ??t?? retir??es des calculs, car la quantit?? des donn??es figurant dans le dataset d???origine n?????tait pas suffisante.



    """)
    st.markdown("")
    st.markdown("")


    st.subheader("South Australia et Western Australia")

    st.markdown("""
    L???entrainement de donn??es a ??t?? fait avec et sans over-sampling. Les 2 dataset ont ??t?? entrain??s s??paremment. Enfin, Nous avons lanc?? un GridCVSearch afin de trouver les meilleurs param??tres.\n
    
    Le dataset South Australia comportait 8853 lignes et celui de Western Australia 23 203 lignes.\n
    
    La Logistic Regression  avec Oversampling donne pratiquement les m??mes r??sultats que Random Forest avec Oversampling et dans ce cas, vaut mieux adopter le mod??le ayant le traitement le plus rapide.\n\n

    """)
    
    st.image("images//modele2.png")
    st.markdown("")
    st.markdown("")
    
    
    
    st.subheader("Tasmanie et Norfolk Island")

    st.markdown("""

    L???entrainement de donn??es a ??t?? fait avec et sans over-sampling. Les 2 dataset ont ??t?? r??unis car les donn??es ??taient peu nombreuses et le dataset final comporte 5905 lignes. Les mod??les ont ??t?? entrain??s sur les 5905 lignes. Enfin, Nous avons lanc?? un GridCVSearch afin de trouver les meilleurs param??tres.

    """)

    st.image("images//modele3.png")
    st.markdown("")
    st.markdown("")



    st.subheader("New South Wales et Victoria")

    st.markdown("""

    Comme mentionn?? dans la section pr??c??dente, les donn??es de la variable cat??gorielle, RainTomorrow, ne sont pas ??quilibr??es. Par cons??quent, cela peut entra??ner des pr??dictions peu fiables sur les donn??es examin??es. Comme indiqu?? pr??c??demment dans la section sur les mod??les de classification des ??les, il a ??t?? d??cid?? de traiter les donn??es par la m??thode du sur??chantillonnage. En outre, compte tenu des pr??visions au niveau national, la m??thode de r??gression logistique ??tait co??teuse en termes de calcul. En combinant ces consid??rations, il a ??t?? d??cid?? que pour les r??gions administratives de Victoria (VIC) et de New South Wales (NSW), les m??thodes de classification ?? employer sont KNN et RandomForest. En outre, notre d??cision est motiv??e par le fait que ces deux r??gions ont le plus grand nombre de stations m??t??orologiques, ce qui pourrait les rendre repr??sentatives de l'ensemble de la base de r??f??rence. Afin d'obtenir une image claire, nous avons d'abord choisi de traiter ces deux r??gions administratives s??par??ment, puis de les traiter ensemble. 

    Le dataset New South Wales comportait xx lignes et celui de Victoria xx lignes
    """)

    st.image("images//modele4.png")
    st.markdown("")
    st.markdown("")



    st.subheader("Synth??se des entrainements")
    
    st.markdown("""
    \n\n
    """)

    st.image("images//synthese.png")



if page ==pages[6]:

    st.header("Restitution graphique des pr??dictions")

    st.markdown("""

    Nous avons choisi Folium pour la simplicit?? de son usage et le rendu graphique.\n

    Nous avons donc r??cup??r?? la longitude et latitude de chaque ville et l???avons stock??e dans une table.\n

    Associ??e ?? la notion de radar, nous repr??senterons les villes radar d???une couleur diff??rente (en rouge).\n

    Nous avons donc construit un tableau unique qui prend en compte les villes, les radars et la variable RainTomorrow. Bien que la variable radar n'apparaisse pas dans la carte g??ographique, il a ??volu?? pour l'inclure afin de faire un double contr??le sur la manipulation correcte du dataframe. La base de donn??es r??sultante a ensuite ??t?? filtr??e afin de comparer les donn??es stock??es dans la trame de donn??es source avec les donn??es pr??dites ?? la m??me date. \n

    Si le temps est beau alors nous figurerons un soleil, sinon une goutte d???eau.\n

    Ci-dessous, une image de la variable RainTomorrow pour la journ??e du 2017/04/2011.



    """)

    st.image("images//graphique1.png")
    st.markdown("")
    st.markdown("")

    st.markdown("""

    Le jeu de base utilis?? est le jeu compl??t??, c'est-??-dire celui dans lequel certaines villes ont ??t?? retir??es parce que la variable RainTomorrow n'a pas ??t?? enregistr??e exp??rimentalement ?? cette date sp??cifique, il n'est pas surprenant que certaines villes ne soient pas pr??sentes sur la carte.\n

    Comme cet exemple de d??monstration est r??alis?? sur la r??gion de New South Wales, la pr??vision pour les villes situ??es dans cette r??gion est pr??sent??e ci-dessous.

    """)

    st.image("images//graphique2.png")
    st.markdown("")
    st.markdown("")



    st.markdown("""

    On observe que le 2017/04/11, il a plu sur le littoral de la r??gion de New South Wales, plus pr??cis??ment autour de la ville de Sidney. En revanche, dans les villes les plus proches (par exemple), la journ??e a ??t?? ensoleill??e. Cela cr??e une zone limite tr??s fine et la pr??cision de la m??thode en question est donc mise ?? l'??preuve.  



    Nous tenons ?? souligner que dans notre m??thode d'apprentissage automatique, la date n'??tait pas indiqu??e dans l'??chantillon X_test. Pour cette raison, une fois que le mod??le a ??t?? entra??n?? et que la variable a ??t?? pr??dite, un cadre de donn??es contenant les pr??dictions a ??t?? construit et ensuite fusionn?? avec le cadre de donn??es X_test par la m??thode de fusion afin d'avoir un cadre de donn??es complet contenant les emplacements. Les variables latitude, longitude et toutes les autres variables pr??sentes dans le cadre de donn??es original, filtr??es par date, ont ensuite ??t?? ajout??es ?? ce cadre de donn??es.



    Voici le r??sultat de la pr??diction sur la r??gion NSW.



    """)

    st.image("images//graphique3.png")
    st.markdown("")
    st.markdown("")

    st.markdown("""

    Comme on peut le voir sur la figure, la variable RainTomorrow le 2017/04/11 n'a ??t?? pr??dite que pour 3 villes (Albury, Canberra, et Richmond). Ceci est coh??rent avec le fait que lors de l'??tape de division de l'ensemble de donn??es en test et formation, l'??chantillon de formation repr??sente 20% de l'ensemble de donn??es.

    """)



if page == pages[5]:

    st.header("Conclusions")

    st.markdown("""

    Ce projet de master s'est concentr?? sur la pr??diction de la variable rainTomorrow, et a permis de mettre en pratique les concepts ??tudi??s dans le cadre de la formation Data Scientest. Afin de pr??dire la variable, des protocoles ont ??t?? d??velopp??s sur la base de l'??tat de l'art en mati??re de calcul. Afin de d??velopper ces protocoles, un traitement du jeu de donn??es a d'abord ??t?? effectu??, suivi d'une analyse statistique et graphique, permettant d'obtenir une vue d'ensemble.\n\n

    Comme les pr??visions m??t??orologiques peuvent suivre des cycles dans le temps, une ??tude des s??ries temporelles a ??t?? entreprise. Cependant, il faut savoir que le manque de m??moire de nos ordinateurs fait planter le traitement. Cela peut ??tre imputable ?? la taille de l'ensemble de donn??es ; ce r??sultat a ??galement ??t?? obtenu dans le cas d'un ensemble de donn??es r??duit (3 ?? 5 ans). Les s??ries temporelles nous auraient permis de pr??voir ?? la fois RainTomorrow mais aussi MaxTemp et MinTemp. 

    """)

    st.image("images//conclusion1.png")
    st.markdown("")

    st.markdown("""

    Nous nous sommes donc content??s d?????valuer les performances des pr??dictions des classifieurs que nous avons test??s. Les meilleurs hyperparam??tres varient parfois selon les r??gions administratives. En revanche, les hyperparam??tres de la m??thode de la for??t al??atoire sont les m??mes pour toutes les r??gions. De plus, cette m??thode s'est av??r??e ??tre la plus rapide (en termes de temps r??el) par rapport aux deux autres m??thodes.

    """)

    st.image("images//conclusion2.png")
    st.markdown("")

    st.markdown("""

    En conclusion, notre ??tude nous a permis ?? la fois de d??terminer la variable cat??gorielle en question avec une bonne pr??cision et de d??terminer les hyperparam??tres optimaux. \n\n

    Enfin, il faut souligner que l'utilisation de la biblioth??que missingo montre que l'ensemble de la base de donn??es est caract??ris?? par un nombre ??lev?? de valeurs manquantes.\n\n


    """)
    
    st.image("images//conclufin.png")
    st.markdown("")

if page == pages[7]:
    st.header("Entrainez nos mod??les !")

    region  = ["New South Wales", "Western Australia", "Autralia"]
    model  = ["KNN", "Random Forest", "Logistic Regression"]
    # overSampl = ["Avec", "Sans"]

    selectionRegion = st.selectbox('Choisissez une r??gion', options = region)
    selectionModel = st.selectbox('Choisissez un mod??le', options = model)
    # over = st.radio("Votre choix OverSampling", options= overSampl)
    over = st.radio("Votre choix OverSampling", ('Avec', 'Sans'))
    
    st.markdown("""
    R??sultats de l'entrainement
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
    
    st.header("Nous allons voir une pr??diction pour le 11 avril 2017")
    
    region=st.selectbox("Choisissez une r??gion", ('New South Wales', 'Western Australia'))
    
    st.markdown("")
    st.markdown("")
    
    
           
    st.write('Pr??diction du temps sur la r??gion ', region, ' avec RandomForest et OverSampling')
       
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
