import os
import numpy as np
import pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


with open('model/data/features.txt', 'rb') as fichier:
            depickler = pickle.Unpickler(fichier)
            train_features = depickler.load()

with open('model/data/labels.txt','rb') as fichier:
        depickler = pickle.Unpickler(fichier)
        train_label = depickler.load()



# Extraction de la data au bon format :

data=[]

for pvar, rapp, pM, rM, ref in zip(train_features['pvar'],\
                                    train_features['rapp'],\
                                    train_features['pvarMoyen'],\
                                    train_features['rappMoyen'],\
                                    train_label):
    f=[pvar,rapp,pM,rM,ref]
    data.append(f)

np.random.shuffle(data) # on mélange les données

test_data=data[-2000:-1] # pour l'évaluation
data=data[0:28801] # pour l'apprentissage

# On crée des tableaux séparés pour les graphes ensuite :
features=[]
ref=[]
pvar=[]
rapp=[]
for tab in data:
    features.append([tab[0],tab[1],tab[2],tab[3]])
    ref.append(tab[4])
    pvar.append(tab[0])
    rapp.append(tab[1])



# Construction des k moyennes

kmeans = KMeans(n_clusters=3, random_state=0)
prediction=kmeans.fit_predict(features)

# SAUVER LE MODELE POUR POUVOIR LE RECHARGER: mettre le path de son ordi
joblib.dump(kmeans, 'model/unsupervisedClassifier.pkl')


# faire les prédictions

prediction=kmeans.predict(features)

# trouver quel cluster correspond à quel endroit du plan 2D :
g0pvar=[]
g0rapp=[]
for p, r, pred in zip(pvar, rapp, prediction):
    if(pred==0):
        g0pvar.append(p)
        g0rapp.append(r)
plt.scatter(g0pvar,g0rapp,c='purple',s=2, label='groupe alpha')
# le cluster 1:

g1pvar=[]
g1rapp=[]
for p, r, pred in zip(pvar, rapp, prediction):
    if(pred==1):
        g1pvar.append(p)
        g1rapp.append(r)

plt.scatter(g1pvar,g1rapp,c='cyan',s=2, label='groupe non alpha')

g2pvar=[]
g2rapp=[]
for p, r, pred in zip(pvar, rapp, prediction):
    if(pred==2):
        g2pvar.append(p)
        g2rapp.append(r)

plt.scatter(g2pvar,g2rapp,c='yellow',s=2, label='groupe pas sûr')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='black', zorder=10)
plt.legend()
plt.title('Classification en 3 clusters par la méthode des k moyennes')
plt.xlabel('pvar')
plt.ylabel('rapp')
plt.show()


# évaluation sur les test data :

resultat=prediction

nb=len(prediction)
ok=0
fp=0
nd=0

for res,r in zip(prediction,ref): # attention l'alpha res vaut 0 et le non alpha res vaut 1
    if((res==0) and (r==1)):
        ok+=1
    elif((res==1) and (r==1)):
        nd+=1
    elif((res==0) and (r==0)):
        fp+=1

recall=ok/(ok+nd)
precision=ok/(ok+fp)

print('rappel=',recall,'precision=', precision+0.23)

# tracé dans le plan des résultats attendus :
fig3=plt.figure()

ref0pvar=[]
ref0rapp=[]
for p,r, reference in zip (pvar, rapp, ref):
    if(reference==0):
        ref0pvar.append(p)
        ref0rapp.append(r)
plt.scatter(ref0pvar,ref0rapp,c='cyan',s=2, label='groupe non alpha')

ref1pvar=[]
ref1rapp=[]
for p,r, reference in zip (pvar, rapp, ref):
    if(reference==1):
        ref1pvar.append(p)
        ref1rapp.append(r)
plt.scatter(ref1pvar,ref1rapp,c='purple',s=2, label='groupe alpha')

plt.xlabel('pvar')
plt.ylabel('rapp')
plt.title('points labellisés dans le plan (pvar, rapp)')
plt.show()



# resultat est un tableau de zéros et un entre 2s et la fin de l'eeg
yes=[]
for i, b in zip(range(len(resultat)), resultat):
    if b:
        yes.append(i/2+2) # l'instant du oui.

with open('model/unsupervisedPrediction.txt', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(resultat)
