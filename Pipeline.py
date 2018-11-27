import os
import numpy as np
import mne
from sklearn import linear_model
from sklearn.externals import joblib

''' Ce programme contient:
    - des fonctions intermédiaires
    - une fonction traitementfinal qui prend en argument raw, start, end
    et renvoie un tableau de bool donnant la prédiction.

Il charge un modèle scikit déjà entrainé qui doit être sauvé dans
un fichier dans l'ordi. IL FAUT METTRE L ADRESSE ET LE NOM DE CE FICHIER
LIGNE 190-191. '''


# FONCTIONS AUXILIAIRES POUR LA FONCTION DE PREDICTION :

def convert(yes,end):
    '''extrait l'information du tableau d'intervalles
    pour donner un tableau de zéros et de un toutes les demi secondes, entre
    start et end '''
    yesno=np.zeros(2*int(end),bool) # un point de yesno pour une demiseconde
    for intervalle in yes:
        i = int(2*intervalle[0])
        while((i<int(2*intervalle[1])) and (i<len(yesno))):
            yesno[i]=True
            i+=1
    return yesno

def cut(yesno, start, end):
    ''' coupe le yesno produit ci dessus à start et à end'''
    yesno = yesno[2*start :2*end]
    return yesno

def variation(tab , i) :
    return (tab[i+1]-tab[i])

def derivativeArray(tab, temps, sig):
    """prend en argument un tableau de valeurs  et le
    tableau de temps, et renvoie un tableau contenant les valeurs absolues des
    pentes du rapport à chaque instant."""
    derivativeArray=[0 for i in range(len(tab))]

    if(len(tab)!=len(temps)):
        print("pb de dimension, len(tab)=" + str(len(tab)) + "len(temps)=" +\
              str(len(temps)))
        return;
    for i in range(len(tab)-1):
        derivativeArray[i]=np.abs(variation(tab,i))/variation(temps,i)


    return derivativeArray

def maxa1(sig, fres, tab, infa, supa):
    '''tab : tableau des instants de temps
        sig : le signal
        fres : la résolution en fréquence du tableau des ondelettes
    renvoie un tableau de frequences,  et un tab de puissances
    contenant la fréquence prépondérante dans l'intervalle [infa,supa] et sa
    puissance pour chaque instant'''
    maxp = [0 for i in range(len(tab))] # les puissances des fmax
    maxfreq = [0 for i in range(len(tab))] # les fmax
    for t in range(len(tab)):

        maxfreq[t] = np.argmax(tab[t][infa*fres:supa*fres])/fres + infa
        maxp[t] = np.max(tab[t][infa*fres:supa*fres])
    return maxfreq,maxp

def ondelette(sig, fn, fres, sfreq, func, vals,dur ):

    tab = [[0 for f in range(fn)] for i in range(len(sig))]
    d = int(dur*sfreq/2) # la demi longueur de la fenêtre de produit
    #d'ondelettes, en nombre de points.

    for f in range(fn): # pour chaque fréquence

        v = func(f/fres,len(vals)//(2*sfreq)) # la fonction ondelette centrée
        v = v[len(v)//2 - d :len(v)//2 + d] # coupe le tableau symétriquement
        #autour de du milieu, en enleant d de chaque côté.


        for t in range(d, len(sig)-d): #calcul des coefficients P(f,t) du
            #tableau d'ondelettes.
            #print(len(v))
            #print(len(sig[t-int(dur*sfreq/2):t+int(dur*sfreq/2)]))

            s = np.dot(sig[t - d:t + d], v) # amplitude complexe en t,f
            tab[t][f] = np.real(s)**2 + np.imag(s)**2
    return tab[d:len(sig)-d][:]



def trait(raw,start, end, channel):
    '''raw: l'eeg raw préchargé
    start, end: instants bornes du traitement
    channel : liste des channels sur lesquels on calcule.
    infa, supa : les bornes de la bande de fréquences. '''

    nombrepointsparsec = 2 # un entier
    dureefenetre = 2 # la taille de la fenetre glissante.
    dur=1 # la durée sur laquelle on fait le produit d'ondelette
    nb=nombrepointsparsec*int(end-start) #nombre de points où l'on calcule les
    #descripteurs
    infa=6 # la borne inf des fréquences alpha considérées
    supa=13  # la borne sup des fréquences alpha considérées

    channel=[channel]
    specific_chans = raw.copy().pick_channels(channel)
    flo=5
    fhi=30
    sig=specific_chans.filter(flo, fhi)  #filtrage du signal entre flo et fhi

    sfreq=sig.info['sfreq'] # fréquence d'échantillonnage, ici 256Hz donnée

    a =  int(sfreq * (start - dur/2))
    # pour avoir un tableau résultant entre start et end, il faut regarder
    # le signal entre start-dur/2 et end+dur/2
    b =  int(sfreq * (end + dur/2 ))
    data, times = sig[0, a:b] #restriciton temporelle du signal

    sig=10**3*data[0]

    fmax = fhi # fréquence maximale à visualiser
    fres = 2 # résolution en fréq (exp: 0.25 Hz pou fres = 4)
    # 1/fres est donc l'écart entre 2 fréquences dans le tableau ondelettes

    fn = fmax*fres # le nombre de fréquences dans le tableau d'ondelettes

    vals = np.linspace(0, end - start, (end-start)*sfreq)
    # le tableau de temps va de 0 à end-start, avec le meme nb de points que
    # le signal

    # freqs = np.linspace(0, fmax, fmax*fres) #gamme des fréquences

    func = lambda f,t : 10000000*(np.exp(1j*2*np.pi*(f)*(vals-t)) - \
                                  np.exp(-0.5*(2*np.pi*f)**2)) * \
                                  np.exp(-0.34*((vals-t)**2)*(f+1)**2)


    tab = ondelette(sig, fn, fres, sfreq, func,vals, dur )
    # tab est donc un tableau de longueur end-start-2*dur/2

    maxa = maxa1(sig,fres,tab,infa,supa) # fréquence alpha prépondérante
    maxafreq=maxa[0] # la fréquence
    maxap=maxa[1] # la puissance associée

    maxt = maxa1(sig,fres,tab,5,8) # fréquence theta prépondérante

    maxa2=[x**2 for x in maxafreq] #carre des frequences
    dmaxa2=derivativeArray(maxafreq,vals, sig) # variation de la fréquence
    dmaxa2=[x**2 for x in dmaxa2] #carre des derivees

    pvara = []
    alphasurtheta=[]

    delta = int(sfreq * ( end - start) / nb)
    # l'écart entre 2 points sur le graphe final, en nombre de valeurs dans le
    #tableau t (d'où la multiplication par sfreq)

    #print (delta)
    #print (len(maxa2))

    for i in range(nb): # pour chaque point du tableau final
        k=int(i*delta)
        s1 = np.sum(maxa2[k:k+int(sfreq*dureefenetre)])
        s2 = np.sum(dmaxa2[k:k+int(sfreq*dureefenetre)])
        powa= np.sum(maxap[k:k+int(sfreq*dureefenetre)])
        powt = np.sum(maxt[1][k:k+int(sfreq*dureefenetre)])

        pvara.append(s2/s1)
        alphasurtheta.append(powa/powt)

    preprocessing ={ 'start': start,\
                    'end': end,\
                    'pvar': np.array(pvara),
                    'rapp': np.array(alphasurtheta) }

    return preprocessing


def traitementfinal(raw,start,end):
    """prend en argument l'objet raw créé par mne, un instant start >2sec
    et un instant de fin < fin_eeg-2, et renvoie un tableau de bool avec
    2*int(end-start) valeurs (un point par demi-seconde) de prédictions. """

    chan_list= ['EEG O1'] # rajouter les autres channels après
    resultat={} # le dictionnaire 'channel'/tableau de bools renvoyé à la fin

    # importer le modèle : METTRE LA BONNE ADRESSE ET NOM DE FICHIER !
    classifier = joblib.load('supervisedClassifier.pkl')

    # POUR CHAQUE CHANNEL : CALCUL DE LA PREDICTION :

    for channel in chan_list:
        preprocessing=trait(raw,start, end, channel)

        # création du dictionnaire test_features pour la prédiction :

        start= preprocessing['start']
        end= preprocessing['end']

        Ep=np.mean(preprocessing['pvar'])
        Er=np.mean(preprocessing['rapp'])
        taille=len(preprocessing['pvar'])
        moyennep, moyenner = np.zeros(taille), np.zeros(taille)

        for i in range(taille):
            moyennep[i]=Ep
            moyenner[i]=Er

        pmean=np.array(moyennep)
        rmean=np.array(moyenner)

        # le dictionnaire de test features :
        test_features={'pvar': preprocessing['pvar'],\
                       'rapp': preprocessing['rapp'],\
                       'pvarMoyen': pmean,\
                       'rappMoyen': rmean}


        prediction=[]

        for pvar, rapp, pvarM, rappM in zip(test_features['pvar'],\
                                    test_features['rapp'],\
                                    test_features['pvarMoyen'],\
                                    test_features['rappMoyen']):
            f=[pvar,rapp,pvarM,rappM]
            prediction.append(classifier.predict([f])[0])

        print(prediction)

        resultat[channel]=prediction

    return resultat['EEG O1'];
'''comme on s etait mis d accord avec Milan pour que la fonction renvoie
un tableau de bool, je renvoie la prédiction pour l'électrode 001 ici,
mais il y a moyen de renvoyer les 3 prédictions, ou de croiser les informations
des 3 pour renvoyer un seul tableau de bool plus fiable, à discuter '''
