import os
import numpy as np
import pickle
from sklearn import svm, linear_model
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt



with open('model/data/features.txt', 'rb') as fichier:
            depickler = pickle.Unpickler(fichier)
            train_features = depickler.load()

with open('model/data/labels.txt','rb') as fichier:
        depickler = pickle.Unpickler(fichier)
        train_label = depickler.load()



# Build sklearn SVM classifier :

classifier = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0,\
                      shrinking=True, probability=False, tol=0.001, \
                      cache_size=300, class_weight={1:2}, verbose=False, \
                      max_iter=-1, decision_function_shape='ovr', \
                      random_state=None)

# ou lineaire, avec des resultats moins bons
"""
classifier = linear_model.SGDClassifier(alpha=0.0001, average=False, \
                                        class_weight={1: 2}, epsilon=0.001,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=True)
"""

data=[]

for pvar, rapp, pM, rM, ref in zip(train_features['pvar'],\
                                    train_features['rapp'],\
                                    train_features['pvarMoyen'],\
                                    train_features['rappMoyen'],\
                                    train_label):
    f=[pvar,rapp,pM, rM, ref]
    data.append(f)

np.random.shuffle(data) # on mélange les données

test_data=data[-2000:-1] # pour l'évaluation
data=data[0:-2000] # pour l'apprentissage

features=[]
ref=[]
pvar=[]
rapp=[]
for tab in data:
    features.append([tab[0],tab[1],tab[2],tab[3]])
    ref.append(tab[4])

test_f=[]
test_r=[]
for tab in test_data:
    test_f.append([tab[0],tab[1],tab[2],tab[3]])
    test_r.append(tab[4])


classifier.fit(features,ref)


# évaluation sur les test data :
prediction=[]

for data in test_f:
    prediction.append(classifier.predict([data])[0])

'''print(prediction)'''
resultat=prediction

nb=len(prediction)
ok=0
fp=0
nd=0
for res,ref in zip(prediction,test_r):
    if((res==1) and (ref==1)):
        ok+=1
    elif((res==0) and (ref==1)):
        nd+=1
    elif((res==1) and (ref==0)):
        fp+=1

recall=ok/(ok+nd)
precision=ok/(ok+fp)

print('rappel=',recall,'precision=', precision)

# SAUVER LE MODELE POUR POUVOIR LE RECHARGER: mettre le path de son ordi
joblib.dump(classifier, 'model/supervisedClassifier.pkl')


# resultat est un tableau de zéros et un entre 2s et la fin de l'eeg
yes=[]
for i, b in zip(range(len(resultat)), resultat):
    if b:
        yes.append(i/2+2) # l'instant du oui.

with open('model/supervisedPrediction.txt', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(resultat)
