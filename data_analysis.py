# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:57:51 2022

@author: aberg
"""
# Modules nécessaires
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.metrics import f1_score
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression

# Préparation de la data
ds_subset = pd.read_csv("data/spotify_dataset_subset.csv")
ACP=pd.read_csv("data/spotify_dataset_subset.csv")
print("Taille du dataset_train :", np.shape(ds_subset))

y = ds_subset.iloc[0:500000, [5]]
y = y.to_numpy()
ds_subset = ds_subset.drop(columns=['popularity', 'artist_name', 'track_name', 'id', 'genres'])
ds_subset.info()
ds_subset = ds_subset.to_numpy()
for row in ds_subset:
    if len(row[0]) > 7:
        row[0] = datetime.strptime(row[0], "%Y-%m-%d").year
    elif len(row[0]) > 4:
        row[0] = datetime.strptime(row[0], "%Y-%m").year
    else:
        row[0] = datetime.strptime(row[0], "%Y").year

    if row[1]:
        row[1] = 1
    else:
        row[1] = 0

ACP = ACP.drop(columns=['artist_name', 'track_name', 'id', 'genres'])
ACP.info()
ACP = ACP.to_numpy()
for row in ACP :
    if len(row[0]) > 7:
        row[0] = datetime.strptime(row[0], "%Y-%m-%d").year
    elif len(row[0]) > 4:
        row[0] = datetime.strptime(row[0], "%Y-%m").year
    else:
        row[0] = datetime.strptime(row[0], "%Y").year

    if row[1]:
        row[1] = 1
    else:
        row[1] = 0

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.cm as cm

import seaborn as sns

from sklearn.decomposition import PCA

def biplot(pca=[],x=None,y=None,components=[0,1],score=None,coeff=None,coeff_labels=None,score_labels=None,circle='T',bigdata=1000,cat=None,cmap="viridis",density=True):

    if isinstance(pca,PCA)==True :

        coeff = np.transpose(pca.components_[components, :])

        score=  pca.fit_transform(x)[:,components]

        if isinstance(x,pd.DataFrame)==True :

            coeff_labels = list(x.columns)

    if score is not None : x = score

    if x.shape[1]>1 :

        xs = x[:,0]

        ys = x[:,1]

    else :

        xs = x

        ys = y

    if (len(xs) != len(ys)) : print("Warning ! x et y n'ont pas la même taille !")

    scalex = 1.0/(xs.max() - xs.min())

    scaley = 1.0/(ys.max() - ys.min())

    #x_c = xs * scalex

    #y_c = ys * scaley

    temp = (xs - xs.min())

    x_c = temp / temp.max() * 2 - 1

    temp = (ys - ys.min())

    y_c = temp / temp.max() * 2 - 1

    data = pd.DataFrame({"x_c":x_c,"y_c":y_c})

    print("Attention : pour des facilités d'affichage, les données sont centrées-réduites")

    if cat is None : cat = [0]*len(xs)

    elif len(pd.Series(cat)) == 1 : cat = list(pd.Series(cat))*len(xs)

    elif len(pd.Series(cat)) != len(xs) : print("Warning ! Nombre anormal de catégories !")

    cat = pd.Series(cat).astype("category")

    fig = plt.figure(figsize=(6,6),facecolor='w') 

    ax = fig.add_subplot(111)

    # Affichage des points

    if (len(xs) < bigdata) :   

        ax.scatter(x_c,y_c, c = cat.cat.codes,cmap=cmap)

        if density==True : print("Warning ! Le mode density actif n'apparait que si BigData est paramétré.")

    # Affichage des nappes convexes (BigData)

    else :

        #color

        norm = mpl.colors.Normalize(vmin=0, vmax=(len(np.unique(cat.cat.codes)))) #-(len(np.unique(c)))

        cmap = cmap

        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        if density==True :

            sns.set_style("white")

            sns.kdeplot(x="x_c",y="y_c",data=data)

            if len(np.unique(cat)) <= 1 :

                sns.kdeplot(x="x_c",y="y_c",data=data, cmap="Blues", shade=True, thresh= 0)

            else :

                for i in np.unique(cat) :

                    color_temp = m.to_rgba(i)

                    sns.kdeplot(x="x_c",y="y_c",data=data[cat==i], color=color_temp,

                                shade=True, thresh=0.25, alpha=0.25)     

        for cat_temp in cat.cat.codes.unique() :

            x_c_temp = [x_c[i] for i in range(len(x_c)) if (cat.cat.codes[i] == cat_temp)]

            y_c_temp = [y_c[i] for i in range(len(y_c)) if (cat.cat.codes[i] == cat_temp)]

            points = [ [ None ] * len(x_c_temp) ] * 2

            points = np.array(points)

            points = points.reshape(len(x_c_temp),2)

            points[:,0] = x_c_temp

            points[:,1] = y_c_temp

            hull = ConvexHull(points)

            temp = 0

            for simplex in hull.simplices:

                color_temp = m.to_rgba(cat_temp)

                plt.plot(points[simplex, 0], points[simplex, 1],color=color_temp)#, linestyle='dashed')#linewidth=2,color=cat)

                if (temp == 0) :

                     plt.xlim(-1,1)

                     plt.ylim(-1,1)

                     temp = temp+1

    if coeff is not None :

        if (circle == 'T') :

            x_circle = np.linspace(-1, 1, 100)

            y_circle = np.linspace(-1, 1, 100)

            X, Y = np.meshgrid(x_circle,y_circle)

            F = X**2 + Y**2 - 1.0

            #fig, ax = plt.subplots()

            plt.contour(X,Y,F,[0])

        n = coeff.shape[0]

        for i in range(n):

            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5,

                      head_width=0.05, head_length=0.05)

            if coeff_labels is None:

                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')

            else:

                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, coeff_labels[i], color = 'g', ha = 'center', va = 'center')

        if score_labels is not None :

            for i in range(len(score_labels)) :

                temp_x = xs[i] * scalex

                temp_y = ys[i] * scaley

                plt.text(temp_x,temp_y,list(score_labels)[i])

    plt.xlim(-1.2,1.2)

    plt.ylim(-1.2,1.2)

    plt.xlabel("PC{}".format(1))

    plt.ylabel("PC{}".format(2))

    plt.grid(linestyle='--')

    plt.show()



# Visualisation de la data
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            fig = plt.figure(figsize=(7, 6))

            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1],
                            X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i],
                             fontsize='14', ha='center', va='center')

            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title(
                "Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks:  # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7, 6))

            # détermination des limites du graphique
            if lims is not None:
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30:
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else:
                xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(
                    pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2, :])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30:
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                           pcs[d1, :], pcs[d2, :],
                           angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0, 0], [x, y]] for x, y in pcs[[d1, d2]].T]
                ax.add_collection(LineCollection(
                    lines, axes=ax, alpha=.1, color='black'))

            # affichage des noms des variables
            if labels is not None:
                for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                        plt.text(x, y, labels[i], fontsize='14', ha='center',
                                 va='center', rotation=label_rotation, color="blue", alpha=0.5)

            # affichage du cercle
            circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)

            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(
                d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
            plt.ylabel('F{} ({}%)'.format(
                d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


scaler = StandardScaler()
scaler.fit(ACP)
scalledData = scaler.transform(ACP)
# ACP
# On paramètre ici pour ne garder que 3 composantes
mypca = PCA(n_components=2)
mypca.fit(scalledData)
res_ACP = mypca.fit_transform(scalledData)
composantes = mypca.components_
nbre_composantes = mypca.n_components
#display_factorial_planes(res_ACP, 2, mypca, [(0,1),(2,3)], labels = None)
features = ["release_date", "explicit","popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "tempo", "time_signature"]
display_circles(composantes, nbre_composantes, mypca, axis_ranks=[
                (0, 1), (1, 2)], labels=np.array(features))
biplot(mypca,x=scalledData,cat=y,components=[0,1])

plt.show()


ds_train = []
y_train = []
ds_prediction = []
y_prediction = []
for i in range((len(y))):
    if (y[i] != 0):
        ds_train.append(ds_subset[i])
        y_train.append(y[i])
    else:
        ds_prediction.append(ds_subset[i])


ds_train = np.array(ds_train)
y_train = np.array(y_train)
ds_prediction_ = np.array(ds_prediction)
y_prediction = np.array(y_prediction)

print(ds_train[0])
print(y_train[0])
print(ds_prediction[0])
print(len(y_train))
print(len(ds_train[0]))

print(y[0])
print(y[1])
print(len(y))
reg = LinearRegression().fit(ds_train, y_train)
print(reg.score(ds_train, y_train))

reg.coef_

x = []
y_train_vis = []
pred = reg.predict(ds_train)
pred_vis = []
for i in range(10):
    print(pred[i][0])
    pred_vis.append(pred[i][0])
    x.append(i)
    y_train_vis.append(y_train[i])
'''
plt.scatter(x, pred_vis, label="Prédiction")
plt.scatter(x, y_train_vis, label="Valeur réelle")
plt.legend()
plt.title("Visualisation de la régression linéaire pour un extrait de 10 chansons du dataset")
plt.show()
'''