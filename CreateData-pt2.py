
import os
import pickle
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

def convert(yes,end):
    '''extrait l'information du tableau d'intervalles
    pour donner un tableau de zeros et de un toutes les demi secondes, entre
    start et end '''
    yesno=np.zeros(2*int(end),bool) # un point de yesno pour une demiseconde
    for intervalle in yes:
        i = int(2*intervalle[0])
        while((i<int(2*intervalle[1])) and (i<len(yesno))):
            yesno[i]=True
            i+=1
    return yesno

def cut(yesno, start, end):
    ''' coupe le yesno produit ci dessus a start et a end'''
    yesno = yesno[2*start :2*end]
    return yesno

# construction du dictionnaire features et du label


liste =['eeg001','eeg002','eeg003','eeg005','eeg009','eeg010','eeg012',\
        'eeg013','eeg014','eeg017',\
        'eeg021','eeg022','eeg023']

for eeg in liste :

    # recuperer les objets des fichier donnees : pour un eeg

    filename ='eegs/annot/res/res_'+eeg+'.txt'
    refname='eegs/annot/ref/reference_'+eeg+'.txt'
    with open(filename, 'rb') as fichier:
            depickler = pickle.Unpickler(fichier)
            reseeg = depickler.load()

    with open(refname,'rb') as fichier:
            depickler = pickle.Unpickler(fichier)
            ref = depickler.load()

    start= reseeg[1]
    end= reseeg[2]
    ref=convert(ref,end)
    ref=np.array(cut(ref,start,end))

    Ep=np.mean(reseeg[0][0])
    Er=np.mean(reseeg[0][1])
    taille=len(reseeg[0][0])
    if (taille != len(ref)):
        print('pb de taille')
    moyennep, moyenner = np.zeros(taille), np.zeros(taille)
    for i in range(taille):
        moyennep[i]=Ep
        moyenner[i]=Er
    if(eeg=='eeg001'):
        label=np.array(ref,dtype=int)
        pvars=reseeg[0][0]
        rapps=reseeg[0][1]
        pmean=np.array(moyennep)
        rmean=np.array(moyenner)
    else :
        label=np.concatenate((label,np.array(ref,dtype=int)))
        pvars= np.concatenate((pvars,reseeg[0][0]))
        rapps= np.concatenate((rapps,reseeg[0][1]))
        pmean=np.concatenate((pmean,moyennep))
        rmean=np.concatenate((rmean,moyenner))


# construction du dictionnaire features et du label

features = {'pvar': np.array(pvars),\
            'rapp': np.array(rapps),\
            'pvarMoyen': np.array(pmean),\
             'rappMoyen': np.array(rmean)}
labels=label
print(len(labels))

with open('model/data/labels.txt', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(labels)
with open('model/data/features.txt', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(features)

print(len(labels))

# data to test the model : un seul eeg.

eeg='eeg023'

filename ='eegs/annot/res/res_'+eeg+'.txt'
refname='eegs/annot/ref/reference_'+eeg+'.txt'
with open(filename, 'rb') as fichier:
        depickler = pickle.Unpickler(fichier)
        reseeg = depickler.load()

with open(refname,'rb') as fichier:
        depickler = pickle.Unpickler(fichier)
        ref = depickler.load()

print(ref)

start= reseeg[1]
end= reseeg[2]
print(start)
print(end)
ref=convert(ref,end)
ref=np.array(cut(ref,start,end),dtype=int)

Ep=np.mean(reseeg[0][0])
Er=np.mean(reseeg[0][1])
taille=len(reseeg[0][0])
if (taille != len(ref)):
    print('pb de taille')
moyennep, moyenner = np.zeros(taille), np.zeros(taille)
for i in range(taille):
    moyennep[i]=Ep
    moyenner[i]=Er

label=ref
pvars=reseeg[0][0]
rapps=reseeg[0][1]
pmean=np.array(moyennep)
rmean=np.array(moyenner)

features = {'pvar': np.array(pvars),\
            'rapp': np.array(rapps),\
            'pvarMoyen': np.array(pmean),\
             'rappMoyen': np.array(rmean)}
labels=label


with open('model/data/test_labels.txt', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(labels)
with open('model/data/test_features.txt', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(features)
