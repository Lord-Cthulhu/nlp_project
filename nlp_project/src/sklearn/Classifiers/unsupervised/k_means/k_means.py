#Dataframe
import numpy as np
import pandas as pd

#Silhouette et Elbow
from sklearn.metrics import silhouette_score, adjusted_rand_score
from kneed import KneeLocator

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split

#Scaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns


import spacy


import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


import time



def excel_to_df(filepath):
    '''
    [Extraction des données excel en df]
    
    [Spécifier:]
    [filepath = Localisation du fichier "data.csv"]
    '''
    try: 
        try:
            df = pd.read_csv(filepath)
            print('csv passed')
            return df
        except Exception as e:
            #print('csv failed')
            #print(f'error: {e}'.format(e))
            pass
        try:
            df = pd.read_excel(filepath)
            print('xlsx passed')
            return df
        except Exception as e:
            #print('xlsx failed')
            #print(f'error: {e}'.format(e))
            pass    
    except Exception as e:
        print('excel_to_df failed')
        print(f'error: {e}'.format(e))
        pass
    
def binary(df,col, x1, x0):
    '''
    [Transormation de données en données binaire]
    
    [Spécifier:]
    [df = Dataframe]
    [col = colonne de référence]
    [x1 = le nom de la varible qui prendra la valeur 1  "val_1"]
    [x2 = le nom de la variable qui prendra la valeur 0 "val_0"]
    '''
    try:
        df[col] = df[col].apply(lambda x: 1 if x== x1
                                  else 0 if x== x0
                                  else None)
        return df 
    except Exception as e:
        print('binary fail')
        print(f'error: {e}'.format(e))
        pass   
    
def cat_binary(df,id,col):
    '''
    [Transormation de données catégoriques en colonnes de valeurs binaires]
    
    [Spécifier:]
    [df = Dataframe]
    [col = colonne de référence]
    [ID = le nom de la colonne ID 'customerID']
    '''
    try:
        #Extraction des données 
        extract_data=df[[id,col]]
        
        #Binarisation avec la fonction dummies
        binarise = pd.get_dummies(extract_data[col],prefix=col) 
        
        #Dataframe contenant le id et les valeurs binarisé
        extract_data2 = pd.concat([extract_data, binarise], axis=1) 
        
        #Retrait de la colonne de référence pour intégrer les variable dummy
        extract_data2.drop(columns=[col],inplace=True)
        
        #On enleve la colonne de référence du jeu de données
        df.drop(columns=[col],inplace=True)
        
        #Joindre les 2 dataframe
        df= pd.merge(df, extract_data2, left_on=id, right_on=id, how='inner')
        
        #On retourne de dataframe un fois les manip terminé
        return df
  #En cas d'erreur on affiche le script qui cause problème et son erreur
    except Exception as e:
        print('cat_binary fail')
        print(f'error: {e}'.format(e))
        pass     
 
def kmeans_stdscaler(f_df, headers):
    '''
    [Scaler avec la méthode standard]
    [f_df = Données propre]
    [headers = le nom des colonnes contenant des variables numériques]

    [return scaled_df]
    '''
    try:
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(f_df) # return an array
        scaled_df = pd.DataFrame(scaled_df, columns=headers)
        return scaled_df
    except Exception as e:
        print('kmeans_stdscaler Fail')
        print(e)
        pass
    
#Détection des outliers basé sur l'écart type (+- 3 ecart type )
def outliers(df,col):
    '''
    [Transormation de données catégoriques en colonnes de valeurs binaires]
    
    [Spécifier:]
    [df = Dataframe]
    [col = colonne de référence]
    '''
    try:
        #Liste vide pour mettre les outliers
        outliers =  []
        
        #Moyenne pour la colonne
        mean = df[col].mean()
        
        #Écart type pour la colonne
        stddev = df[col].std()
        
        #Pour chaque éléments dans la colonne on détecte les outliers et les ajoute a la liste 
        for i in df[col]:
            if i >= mean + 3*stddev:
                outliers.append(i)
            elif i <= mean - 3*stddev:
                outliers.append(i)
        
        #Le nombre de outliers détecté dans la colonne
        c_outliers = len(outliers)
        #print(c_outliers)

        #On génère un Dataframe sans les outliers en itérant le retrait des valeurs nulles 
        x=0
        while x<c_outliers:
            df = df[df[col] != outliers[x]]
            x+=1
        
        #On retourne le nouveau Dataframe
        return df
    
    #En cas d'erreur on affiche le script qui cause problème et son erreur
    except Exception as e:
        print('outliers fail')
        print(f'error: {e}'.format(e))
        pass    
    
def kmeans_preprocessing(df):
    '''
    [Manipulation des données pour apriori]
    [!!ATTENTION!! doit être configuré en fonction du jeu de données]
    [Spécifier:]
    [df = dataframe de référence]
    '''
    try:
        #Retirer les valeurs dupliquées
        #df = df.drop_duplicates(subset=['customerID'], keep=False)
        
        #Inspection des données
        print(df.describe())
        
        
        #Generer des colonnes a partir des ratios
        ##Sessions * Pages / Session = #Pages
        df['Pages'] = df['Sessions'] * df['Pages / Session']
        
        ##Session * Bounce Rate = # Sessions 1 seule page 
        df['1_page'] = df['Sessions'] * df['Bounce Rate']
        
        #Retrait des ratios 
        df = df.drop(columns=['Ecommerce Conversion Rate'])
        df = df.drop(columns=['CPC'])
        df = df.drop(columns=['Bounce Rate'])
        df = df.drop(columns=['Pages / Session'])
        df = df.drop(columns=['Destination URL'])
        
        nlp = spacy.load("en_core_web_sm")
        def tokenize(df):
            try:
                document = nlp(df)
                phrase = []
                for i in document:
                    phrase.append(i.text) 
                return phrase
            except Exception as e:
                print('Tokenize Error')
                print(e)
                pass
        

        # #Retrait des valeurs nulles
        # df.dropna(inplace=True)
        
        # #Remove retrait des espaces
        # #Pas la méthode optimale, ca fonctionne 
        # empty=[]
        # for i in df['TotalCharges']:
        #     if i == " ":
        #         empty.append(i)
        # c_empty = len(empty)
        
        # x=0
        # while x<c_empty:
        #     df = df[df['TotalCharges'] != empty[x]]
        #     x+=1
        #print(df)  
        #df['TotalCharges'] = df['TotalCharges'].astype({'TotalCharges':'float64'})
        #print(df.describe())
        
        # #Transformer les valeur en binaire
        # df = binary(df,'Churn','Yes','No')
        # df = binary(df,'gender','Male','Female')
        # df = binary(df,'Partner','Yes','No')
        # df = binary(df,'Kids','Yes','No')
        # df = binary(df,'PhoneService','Yes','No')
        # df = binary(df,'PaperlessBilling','Yes','No')
        
        #Garder une liste des titres de colonnes pour comparer après manipulation
        df_headers = df.columns.values.tolist()
        print(df_headers)
        print(df)
        

        #print(pd.unique(df['InternetService']))
        
        #df['internet_bin'] = df['InternetService'].apply(lambda x:  1 if x=='DSL' else  1 if x=='Fiber optic' else 0 if x =='No' else 0)
        #print(df['internet_bin'].sum())
        
        # df['StreamingTV'] = df['StreamingTV'].replace(to_replace='No internet service', value='No')
        # df['StreamingMovies'] = df['StreamingMovies'].replace(to_replace='No internet service', value='No')
        # df['MultipleLines'] = df['MultipleLines'].replace(to_replace='No phone service', value='No')
        # df['OnlineSecurity'] = df['OnlineSecurity'].replace(to_replace='No internet service', value='No')
        # df['OnlineBackup'] = df['OnlineBackup'].replace(to_replace='No internet service', value='No')
        # df['DeviceProtection'] = df['DeviceProtection'].replace(to_replace='No internet service', value='No')
        # df['TechSupport'] = df['TechSupport'].replace(to_replace='No internet service', value='No')
        df = outliers(df,'Clicks')
        df = outliers(df,'Revenue')


        # df = binary(df,'StreamingTV','Yes','No')
        # df = binary(df,'StreamingMovies','Yes','No')
        # df = binary(df,'MultipleLines','Yes','No')
        # df = binary(df,'OnlineSecurity','Yes','No')
        # df = binary(df,'OnlineBackup','Yes','No')
        # df = binary(df,'DeviceProtection','Yes','No')
        # df = binary(df,'TechSupport','Yes','No')
        ##################################################################
        
        # #Transformer les variables Cat en dummy 
        # #df = cat_binary(df,'customerID','StreamingTV')
        # #df = cat_binary(df,'customerID','StreamingMovies')
        # #df = cat_binary(df,'customerID','MultipleLines')
        # df = cat_binary(df,'customerID','InternetService')
        # #df = cat_binary(df,'customerID','OnlineSecurity')
        # #df = cat_binary(df,'customerID','OnlineBackup')
        # #df = cat_binary(df,'customerID','DeviceProtection')
        # #df = cat_binary(df,'customerID','TechSupport')
        # df = cat_binary(df,'customerID','Contract')
        # df = cat_binary(df,'customerID','PaymentMethod')
        
        #df.groupby(['Search Query']).sum()
        df = df.groupby(['Search Query'])['Clicks', 'Cost', 'Users', 'Sessions', 'Transactions', 'Revenue', 'Pages', '1_page'].agg('sum')
        df = df.groupby(['Search Query']).first().reset_index()
        
        
        #Ajout des donnees sur keywords
        df['nb_char'] = df['Search Query'].str.len()
        x=0
        listss = []
        for i in df['Search Query']:
            yyy = tokenize(i)
            listss.append(len(yyy))
        df['nb_words'] = listss
        print(df)        
        
        
        
        print(df)

        #print(df['StreamingTV'])
             
        #1b) Select attributes
        # Sélection des valeurs numériques
        num_df = df.select_dtypes(include=np.number)
        # Garder une liste des titres des variables numériques
        headers = num_df.columns.values.tolist()
        
        # Détecter les titres qui ne sont pas utilisés en comparant 2 listes
        rr= [x for x in df_headers if x not in headers]
        print(rr)
        
        #1c) Deal with missing values
        #Rempacer les valeurs manquantes
        f_df = num_df.fillna(num_df.mean())
        
        print(headers)
        print(f_df)
        
        #Selection de la colonne churn 
        s_churn = f_df['Transactions']

        # Pas une étape nécessaire, mais j'ai expérimenté avec downcast signed. Essentiellement je passe de int64 a int8 
        s_churn = pd.to_numeric(s_churn, downcast='signed')
 
        #Variable qui va être réutilisé pour les étiquettes de données
        cat = s_churn
        
        return headers, df, f_df, cat
    except Exception as e:
        print('preprocessing fail')
        print(f'error: {e}'.format(e))
        pass   
#Détection des outliers basé sur l'écart type (+- 3 ecart type )

def outliers(df,col):
    '''
    [Transormation de données catégoriques en colonnes de valeurs binaires]
    
    [Spécifier:]
    [df = Dataframe]
    [col = colonne de référence]
    '''
    try:
        #Liste vide pour mettre les outliers
        outliers =  []
        
        #Moyenne pour la colonne
        mean = df[col].mean()
        
        #Écart type pour la colonne
        stddev = df[col].std()
        
        #Pour chaque éléments dans la colonne on détecte les outliers et les ajoute a la liste 
        for i in df[col]:
            if i >= mean + 3*stddev:
                outliers.append(i)
            elif i <= mean - 3*stddev:
                outliers.append(i)
        
        #Le nombre de outliers détecté dans la colonne
        c_outliers = len(outliers)
        #print(c_outliers)

        #On génère un Dataframe sans les outliers en itérant le retrait des valeurs nulles 
        x=0
        while x<c_outliers:
            df = df[df[col] != outliers[x]]
            x+=1
        
        #On retourne le nouveau Dataframe
        return df
    
    #En cas d'erreur on affiche le script qui cause problème et son erreur
    except Exception as e:
        print('outliers fail')
        print(f'error: {e}'.format(e))
        pass    
        
#Fonction pour utilser la méthode du coude et générer le graphique
def elbow_method(graph,scaled_features,**elbow_kwargs):
    '''
    [elbow_method(graph(1/0),scaled_df,**Kwargs)]
    [graph = Afficher le graphique 1 = oui else = non]
    [scaled_df = Dataframe apres manipulation avec le scaler]
    [kwargs doit contenir les elements suivants]
    "init": "random",
    "n_init": 10,
    "max_iter": 500,
    "random_state": 42,]
        
    [Return opt_cl, sse, runtime]
    '''
    try:
        #Démarrage du timer 
        tic = time.time()

        #A list holds the SSE values for each k
        sse = []
        for i in range(2, 12):
            kmeans = KMeans(n_clusters=i, **elbow_kwargs)
            kmeans.fit(scaled_features)
            sse.append(kmeans.inertia_)
        print('se')
        print(sse)
        if graph == 1: 
            #Affichage du graphique 
            plt.style.use("fivethirtyeight")
            plt.figure(figsize=(10, 10))
            plt.plot(range(2, 12), sse)
            plt.xticks(range(2, 12))
            plt.title('Elbow Method')
            plt.xlabel("Number of Clusters")
            plt.ylabel("SSE")
        else:
            pass
        
        #Détection du nombre optimal de cluster
        kl = KneeLocator(
            range(2, 12), sse, curve="convex", direction="decreasing"
        )
        opt_cl = kl.elbow
        print('elbow nb optimal cluster')
        print(opt_cl)
        print('sse values')
        print(sse)
        
        #Fin du timer
        toc = time.time()
        print("time elbow_method")
        print(round(toc - tic, 1))
        
        #Calcul de la vitesse de traitement
        runtime = round(toc - tic, 1)
        plt.show()
        return opt_cl, sse
    
    except Exception as e:
        print('Failure')
        print(e)
        pass   


def silhouette_method(graph,scaled_features,**silhouette_kwargs):
    '''
        [silhouette_method(visualisation(1/0),scaled_df,**Kwargs)]
        [visualisation = Afficher le graphique 1 = oui else = non]
        [scaled_df = Dataframe apres manipulation avec le scaler]
        [kwargs doit contenir les eleemnts suivants]
            "init": "random",
            "n_init": 10,
            "max_iter": 500,
            "random_state": 42,]
        
        [score, silhouette_coefficients, runtime]
    '''
    try:
        #Démarrage du timer 
        tic = time.time()
        
        
        # A list holds the silhouette coefficients for each k
        silhouette_coefficients = []

        # Notice you start at 2 clusters for silhouette coefficient
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, **silhouette_kwargs)
            kmeans.fit(scaled_features)
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_coefficients.append(score)
            
        #Choisir quand afficher le tableau
        if graph == 1: 
            #Affichage du tableau 
            plt.style.use("fivethirtyeight")
            plt.figure(figsize=(10, 10))
            plt.plot(range(2, 11), silhouette_coefficients)
            plt.xticks(range(2, 11))
            plt.title('Silhouette Method')
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            plt.show()
        else:
            pass
        
        #Performance du modèle 
        print('silhouette score')
        print(score)
        p_max_silhouette = [i for i,x in enumerate(silhouette_coefficients) if x == max(silhouette_coefficients)]
        print('silhouette max value')
        print(max(silhouette_coefficients))
        print(p_max_silhouette)
        print('silhouette_coefficients')
        print(silhouette_coefficients)
        
        #Fin du timer
        toc = time.time()
        print("time silhouette_method")
        print(round(toc - tic, 2))
        
        #Calcul de la vitesse de traitement
        runtime = round(toc - tic, 2)
        
        return [score, silhouette_coefficients, runtime]
    
    except Exception as e:
        print('Failure')
        print(e)
        pass 

def run_kmeans_pipeline(graph,scaled_df,cat,**kmeans_kwargs):
    '''
        [graph = Afficher le graphique 1 = oui else = non]
        [scaled_df = Dataframe apres manipulation avec le scaler]
        [cat = ]
        [kwargs doit contenir les eleemnts suivants]
            "init": "k-means++",
            "n_init": 50,
            "max_iter": 500,
            "random_state": 42,]
    '''
    try:
        # Utiliser la variable n_clusters de la configuration **kwargs pour dynamiser le script
        nb_cluster=kmeans_kwargs.get("n_clusters")

        ##ARI
        # Le ARI n'est pas pertinent dans le contexte
        # ARI assume que je connais les étiquettes réel de données et calcule un indice de similitude entre les valeurs de mon kmeans et les valeurs réels. 
        # Serait utilisé dans dans une approche confirmatoire (Confirmer le classement des cluster et la souche d'un virus)
        # 1= Parfaitement similaire 
        # # Initialisation de l'encodage des étiquettes
        # label_encoder = LabelEncoder()

        # # Définir les étiquettes pour les catégories
        # true_labels = label_encoder.fit_transform(cat)

        # Initialisation du pipeline pour réduire les dimensions
        preprocessor = Pipeline(
            [
                # Pour réduire le nombre de dimensions a 2 (x,y) et permettre de visualiser les clusters
                ("pca", PCA(n_components=2, random_state=1)),
            ]
        )

        # Initialisation du pipeline pour le kmeans
        #Les paramètres **kmeans_kwargs sont référencé au démarrage de la fonction 
        clusterer = Pipeline(
        [
            (   
                "kmeans",
                KMeans(
                    **kmeans_kwargs 
                ),
            ),
        ]
        )

        # Joindre les 2 pipeline 
        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("clusterer", clusterer)
            ]
        )

        # Apprentissage du modèle k-means 
        pipe.fit(scaled_df)
        
        # Joindre les composantes et données stantardisé dans un dataframe
        pcadf = pd.DataFrame(
            pipe["preprocessor"].transform(scaled_df),
            columns=["component_1", "component_2"],
        )
        
        #Liste des clusters prédis par le kmeans
        pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
        cluster = pcadf["predicted_cluster"]

        #Liste des centres des clusters
        centroid = pipe.named_steps["clusterer"]["kmeans"].cluster_centers_
        
        #Performance du modèle K-Means
        inertia = pipe.named_steps["clusterer"]["kmeans"].inertia_ 
        
        #Nombre d'itération avant convergence
        iteration = pipe.named_steps["clusterer"]["kmeans"].n_iter_ 
    
        ###Distance entre les centres 
        #Calculer les distances entres les centres des clusters finaux pour déterminer lesquelques sont les plus dissemblables
        dists = euclidean_distances(centroid)
        print('euclesian distance')
        print(dists)
        
        #ARI
        #Transformer les étiquettes de données à leur encodage d'origine 
        #pcadf["true_label"] = label_encoder.inverse_transform(true_labels) 
        #print('pcadf["true_label"]')
        #print(pcadf["true_label"])
        
        print('centroid', 'inertia','iteration')
        print(centroid, inertia,iteration)

        #Calculer la distance euclédienne entre les points et leur centre de classification
        x_dist = np.linalg.norm(pcadf[['component_1', 'component_2']] - centroid[0,:],axis=1).reshape(-1,1)
        
        #Mettre les distances dans un dataframe
        e_dist = pd.DataFrame(x_dist.sum(axis=1).round(2), columns=["eucledian_distance"])
        
        
        cluster_edist = pd.merge(e_dist, cluster, left_index=True, right_index=True)
        print('cluster_eucledian_distance')
        print(cluster_edist)
        
        #Mettre les statistique des descriptives dans une variable faciliter l'export
        cluster_desc = cluster_edist.groupby(['predicted_cluster']).describe().transpose()
        print('cluster_desc')
        print(cluster_desc)
        
        #Mettre les statistique des fréquences dans une variable faciliter l'export
        cluster_freq = cluster_edist.groupby(['predicted_cluster']).count().transpose()
        print('cluster_freq')
        
        #Simplement pour confirmer que les valeurs sont identiques
        print(cluster_freq)
        
        #Règle pour activer les graphiques
        if graph ==1:
            
            print('centroid')
            print(centroid)
            
            plt.style.use("fivethirtyeight")
            plt.figure(figsize=(10, 10))

            #Mettre les données dans un nuage de points
            scat = sns.scatterplot(
                "component_1",
                "component_2",
                s=50,
                data=pcadf,
                hue="predicted_cluster",
                #style="true_label",
                palette="Set2",
            )
            
            #Mettre les centre dans le nuage de point
            scat = sns.scatterplot(
                centroid[:, 0],
                centroid[:, 1],
                s=50,
                data=centroid,
                color="black",
                markers='x',
            )
            
            #Générer une liste de variable de 0 au nb de cluster 
            label=[] 
            label.extend(range(0, nb_cluster))

            #Sortir les valeurs de la liste en itérant sur la variable label 
            for i,label in enumerate(label):
                
                #Valeur des centroids de l'axe X
                xc = round(centroid[:, 0][i], 3)
                
                #Valeur des centroids de l'axe Y
                yc = round(centroid[:, 1][i], 3)
                
                #On utilise la méthode format pour générer les label de manière itérative 
                c_label = f'Cluster {label}\n (X:{xc},Y:{yc})'.format(label = label, xc = xc, yc = yc)
                
                #Ajout des données au graphique
                plt.annotate(c_label, (centroid[:, 0][i], centroid[:, 1][i]))
            

            #Définition du titre
            scat.set_title(
            "Résultats du K-Means"
            )
            
            #Configuration de la légende
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            
            #Affichage du nuage de points
            plt.show()
            
            ###Distance entre les centres 
            #Définir les dimensions de la figure
            ax = plt.figure(figsize=(10, 10))
            
            #Définir les données à utiliser  dans un heatmap et forcer 3 décimales après le point
            ax = sns.heatmap(dists, annot=True, fmt='.3f')
            
            #Mettre les tackets en haut
            ax.xaxis.tick_top()
            
            #Définir la position des étiquettes
            ax.xaxis.set_label_position('top')
            
            #Définir le titre du tableau
            ax.set_title('Distance entre les centres des clusters')
            
            #Visualiser la figure
            plt.show()
            
            #Définir les dimensions de la figure
            box_plot = plt.figure(figsize=(10, 10))
            
            #Définir les données à utiliser dans un diagramme à moustache
            box_plot = sns.boxplot( x = cluster_edist['predicted_cluster'],y = cluster_edist['eucledian_distance'])
            
            #Définitions des éléments du box_plot pour ajouter des étiquettes de données
            ax = box_plot.axes
            lines = ax.get_lines()
            categories = ax.get_xticks()
            
            #Itération sur les données pour générer les étiquettes
            for icat in categories:
            # À chaque 4 lignes à une intervalle de 6, on retrouve la médiane 
            # p25 = 0 ,p75 = 1, lower_whisker = 2 upper_whisker = 3 p50 = 4  extreme_value = 5 (5 ne fonctionne pas quand certaines colonnes n'ont pas de valeurs extremes)
                
                #Les variables à mettre dans des étiquettes de données
                #Médiane ou P50 (2 dec)
                y_median = round(lines[4+icat*6].get_ydata()[0],2)
                
                #Moustache du bas (2 dec)
                y_lower_whisker = round(lines[2+icat*6].get_ydata()[0],2)
                
                #Moustache du haut (2 dec)
                y_upper_whisker = round(lines[3+icat*6].get_ydata()[0],2)

                #Configuration du texte pour la médiane
                ax.text(
                    icat, 
                    y_median, 
                    f'{y_median}', 
                    ha='center', 
                    va='center', 
                    fontweight='bold', 
                    size=10,
                    color='black',
                    bbox=dict(facecolor='#ffffff')
                    )
                
                #Configuration du texte pour la moustache du bas
                ax.text(
                    icat, 
                    y_lower_whisker, 
                    f'{y_lower_whisker}', 
                    ha='center', 
                    va='center', 
                    fontweight='bold', 
                    size=10,
                    color='black',
                    bbox=dict(facecolor='#ffffff')
                    )
                
                #Configuration du texte pour la moustache du haut
                ax.text(
                    icat, 
                    y_upper_whisker, 
                    f'{y_upper_whisker}', 
                    ha='center', 
                    va='center', 
                    fontweight='bold', 
                    size=10,
                    color='black',
                    bbox=dict(facecolor='#ffffff')
                    )
                
                
            ###############################################################
            #######Méthode alternative fonctionnelle pour les labels#######
            ###############################################################
            
            #Confirme la validité de la méthode précédente 
            
            # medians = cluster_edist.groupby(['predicted_cluster'])['eucledian_distance'].median()
            # vertical_offset = cluster_edist['eucledian_distance'].median() * 0.05 # offset from median for display

            # for xtick in box_plot.get_xticks():
            #     box_plot.text(xtick,medians[xtick] + vertical_offset,medians[xtick], 
            #     horizontalalignment='center',size='x-small',color='w',weight='semibold')
            
            ###############################################################
            
            #Visualiser la figure
            plt.show()  
             
            ###Afficher les centres finaux dans une table
            #Générer une liste des centroids pour mettre au centre du nuage de points
            clist = []
            for i in centroid:
                clist.append(i)
            
            #Configuration du tableau     
            columns = ('x', 'y')
            
            #Méthode dynamique sélectionner les lignes 
            rows = [x for x in list(range(0,nb_cluster))]
            
            #Couleur du tableau basé sur des intervalles
            colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
            
            #Dimensions de la figure
            ax = plt.figure(figsize=(10, 10))
            
            #Retrait des axes pour utiliser une table (pour le look)
            ax = plt.axis('off')
            ax = plt.axis('tight')
            
            #Définition du mode d'affichage et affichage
            ax = plt.table(cellText=clist,
                                rowLabels=rows,
                                rowColours=colors,
                                colLabels=columns,
                                loc='center'
                                )
            
            #Définir le title de la table
            ax = plt.title('Centre finaux')
            
            #Afficher la table
            plt.show()
                
        else:
            #Sinon on affiche pas les graphiques et on sort du if statement 
            pass  
        
        # Liste vide pour garder les scores de performance
        silhouette_scores = []
   
        #ARI
        #ari_scores = []
        
        # itérer entre 2 et le nb de cluster (nb_cluster pour rendre le tout dynamique)
        for n in range(2, nb_cluster):
            # configure le nom de composante pour l'ACP ans influencer les autres étapes
            pipe["preprocessor"]["pca"].n_components = n
            pipe.fit(scaled_df)
            
            
            # Calcule le score pour le coefficient silhouette
            silhouette_coef = silhouette_score(
                pipe["preprocessor"].transform(scaled_df),
                pipe["clusterer"]["kmeans"].labels_,
            )
            # Ajouter le score à la liste 
            silhouette_scores.append(silhouette_coef)  
            
            # ARI
            # # Calcule le score pour ari
            # ari = adjusted_rand_score(
            #     true_labels,
            #     pipe["clusterer"]["kmeans"].labels_,
            # )
            # print('pipe["preprocessor"].transform(scaled_df)')
            # print(pipe["preprocessor"].transform(scaled_df))
            # print('clusterer_labels_')
            # print(pipe["clusterer"]["kmeans"].labels_)
            # print(true_labels)
            # print(true_labels)
            # Ajouter le score à la liste 
            # ari_scores.append(ari)
            
        print('silhouette_coef')
        
        #On regarde le score pour se faire une idée de la validité de la représentation des clusters 
        print(silhouette_coef)
        
        #Règle pour afficher les graphiques
        if graph ==1:
            #Style 
            plt.style.use("fivethirtyeight")
            plt.figure(figsize=(10, 10))
            
            #Mettre les données de performances dans un graphique
            plt.plot(range(2, nb_cluster),silhouette_scores,c="#008fd5",label="Silhouette Coefficient",)
            
            #ARI
            #plt.plot(range(2, nb_cluster), ari_scores, c="#fc4f30", label="ARI")
            
            #Configuration du graphique
            plt.xlabel("n_composants")
            plt.legend()
            plt.title("Performance de regroupement en fonction de n_composants")
            plt.tight_layout()
            
            #Affichage
            plt.show()
        else:
            #Sinon on affiche pas les graphiques et on sort du if statement
            pass    
                
        return cluster
    
    except Exception as e:
        print('run_kmeans_pipeline fail')
        print(f'error: {e}'.format(e))
        pass

def cluster_desc(graph, f_df, cluster):
    try: 
        #Extraire une table des clusters et  faire le compte des valeurs par cluster
        prediction_num = pd.crosstab(index=cluster, 
                                columns="count")
                      
        #Regarder les proportions par cluster
        prediction_num/prediction_num.sum()*100
        print(prediction_num/prediction_num.sum()*100)
        
        #Regarder le décompte par cluster
        print(prediction_num)
        print(cluster)

        #Renommer les noms des clusters
        cluster=cluster.rename({0: 'predicted_cluster'})

        #Joindre la table des clusters à celle des valeurs non standardisées
        print('Joindre les données à des clusters aux données non standardisées')
        df_cnstd= pd.merge(f_df, cluster, how='inner', left_index=True, right_index=True)
        df_cnstd=df_cnstd.rename(index=str)
        print(df_cnstd)

        #Sortir des statistiques descriptives sur les groupes
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        
        #Mettre les statistiques descriptives dans une variable facitiler l'export
        df_desc = df_cnstd.groupby(['predicted_cluster']).describe().transpose()
        
    #     print("df_desc[df_desc[1] == 'gender' & df_desc[2] == 'mean']")
    #     print(df_desc.columns)
    #     df_descx = df_desc[df_desc.iloc[0]]
    #     print(df_descx = df_descx[2])
    #     print(df_desc[df_desc['predicted_cluster'] == 'gender' & df_desc[1] == 'mean'])
        
         
    #     fig, ax = plt.subplots() 
    #     ax.bar(df_desc[1], df_desc.values,  align='center', alpha=0.5, ecolor='black', capsize=10)
    #     ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
    #     ax.set_xticks(df_desc.columns) 
    #    # ax.set_xticklabels(materials)
    #     ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    #     ax.yaxis.grid(True)

    #     # Save the figure and show
    #     plt.tight_layout()
    #     plt.savefig('bar_plot_with_error_bars.png')
    #     plt.show()
        
        #Mettre les statistiques des fréquences dans une variable facitiler l'export
        df_freq = df_cnstd.groupby(['predicted_cluster']).sum().transpose()
        
        
        #l'export dans excel est plus efficace à lire et comprendre 
        if graph ==1:
            ###Graphique var descriptive
            #Configutarion du graphique
            fig, ax = plt.subplots()

            #Cacher les axes
            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            
            #Générer une table avec les valeurs descriptives
            ax.table(cellText=df_desc.values, colLabels=df_desc.columns, rowLabels=list(df_desc.index.values),loc='center')
            fig.tight_layout()

            #Affichage du graphique
            plt.show()
            
            
            ###Graphique fréquence
            #Configutarion du graphique
            fig, ax = plt.subplots()

            #Cacher les axes
            fig.patch.set_visible(False)
            ax.axis('off')
            #ax.axis('tight')
            
            #Générer une table avec les valeurs fréquences
            ax.table(cellText=df_freq.values, colLabels=df_freq.columns,rowLabels=list(df_freq.index.values), loc='center')
            fig.tight_layout()

            #Affichage du graphique
            plt.show()

        else:
            #Sinon on affiche pas les graphiques et on sort du if statement
            pass    


        return df_cnstd,df_desc, df_freq
        
    except Exception as e:
        print('cluster_desc fail')
        print(f'error: {e}'.format(e))
        pass


def anova(df):
    # stats f_oneway functions takes the groups as input and returns ANOVA F and p value
    fvalue, pvalue = stats.f_oneway(df['Clicks'], df['predicted_cluster']) 
    fvalue2, pvalue2 = stats.f_oneway(df['Cost'], df['predicted_cluster']) 
    #, df['Cost'], df['Users'], df['Sessions'] , df['Bounce Rate'] , df['Transactions'] , df['Revenue']
    print('anova')
    print(fvalue, pvalue)
    print(fvalue2, pvalue2)



def cluster_anova(df, header):
    model = ols(header, data=df).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    return aov_table

def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov
#https://www.reneshbedre.com/blog/anova.html
# # output (ANOVA F and p value)
#                  df   sum_sq   mean_sq         F    PR(>F)
# C(treatments)   3.0  3010.95  1003.650  17.49281  0.000026
# Residual       16.0   918.00    57.375       NaN       NaN

path = 'data/gads_data2.xlsx'
df = excel_to_df(path)

########################
#########K means########
########################

#Extraction des données suite au preprocessing
headers, df2 ,f_df, cat = kmeans_preprocessing(df)

print(df2)

df2.to_excel('clean_df3.xlsx')

#Choix du modèle de scaler 
#scaled_df = analytics.kmeans_minmaxscaler(f_df,headers)

#J'utilise le scaler standard
scaled_df = kmeans_stdscaler(f_df,headers)

#Configuration pour les 2 méthodes (elbow, silhouette)
sil_elb_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 500,
    "random_state": 42,
}

#Détection du nombre optimal de cluster avec silhouette
score, silhouette_coefficients, runtime_silhouette = silhouette_method(1,scaled_df,**sil_elb_kwargs)
print(score, silhouette_coefficients, runtime_silhouette)
#times["silhouette"] = runtime_silhouette
print(scaled_df)
#Détection du nombre optimal de cluster avec elbow
opt_cl, sse= elbow_method(1,scaled_df,**sil_elb_kwargs)
#print(opt_cl, sse, runtime_elbow)
#times["elbow"] = runtime_elbow

#Configuration pour le Kmeans
kmeans_kwargs = {
    "n_clusters":4,
    "init": "k-means++",
    "n_init": 50,
    "max_iter": 500,
    "random_state": 42,
}

#Choix du modèle pour rouler l'analyse kmeans

#Avec Pipeline et PCA
#À partir du moment ou le jeu de données contient plus de 3 colonnes, on doit utiliser un pipeline et limiter le nombre de dimensions avec une acp pour pouvoir visualiser les données dans un tableau
cluster = run_kmeans_pipeline(1,scaled_df,cat,**kmeans_kwargs )

#Exporter les informations sur les clusters
df_cnstd, df_desc, df_freq = cluster_desc(0,f_df, cluster)


#aov = cluster_anova(df, headers)
#table = anova_table(aov)
#print(table)

df_cluster = pd.DataFrame()
#df_cluster = df_cnstd

df_cnstd.reset_index(drop=True, inplace=True)
df2['Search Query'].reset_index(drop=True, inplace=True)
print(df2)
df_cluster = pd.concat([df_cnstd,df2['Search Query']], axis = 1)
print('test')
print(df_cluster)
anova(df_cnstd)

#Exporter les tables dans un fichier excel 
with pd.ExcelWriter('cluster_group_test3.xlsx') as writer:
    df_desc.to_excel(writer, sheet_name = "Description")
    df_freq.to_excel(writer, sheet_name = "Frequence")
    df_cluster.to_excel(writer, sheet_name = "Clustered_Data")
