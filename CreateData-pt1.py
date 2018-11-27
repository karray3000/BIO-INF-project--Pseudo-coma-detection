import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.datasets import sample
from mne import create_info, EpochsArray
from mne.baseline import rescale
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)
import os
import pickle

# Variables globales, ne pas toucher.

sfreq=256.0
flo = 5 #fréq de coupure basse
fhi = 30 #fréq de coupure haute


def variation(tab , i) :
    return (tab[i+1]-tab[i])

def derivativeArray(tab, temps, sig):
    """prend en argument un tableau de valeurs (par exemple le rapport de palpha sur pmax) et le
    tableau de temps, et renvoie un tableau contenant les valeurs absolues des pentes du rapport à chaque instant.
    si on veut pas la valeur absolue, enlever le np.abs"""
    derivativeArray=[0 for i in range(len(tab))]

    if(len(tab)!=len(temps)):
        print("pb de dimension, len(tab)=" + str(len(tab)) + "len(temps)=" +str(len(temps)))
        return;
    for i in range(len(tab)-1):
        derivativeArray[i]=np.abs(variation(tab,i))/variation(temps,i)


    return derivativeArray


def var1(tab):# correlation du signal, dispersion frequence autour de leur moyenne
    return np.sqrt(np.var(tab))/np.mean(tab)


def var2(tab): #somme des derivees au carre/ somme des carres, montre les grandes vars temp de frequence
    l = derivativeArray(tab,vals, sig)
    l = [i**2 for i in l]
    l1 = [i**2 for i in tab]
    s = np.sum(l)/np.sum(l1)
    return s


def ondelette(sig, fn, fres, sfreq, func, vals,dur ):

    tab = [[0 for f in range(fn)] for i in range(len(sig))]
    d = int(dur*sfreq/2) # la demi longueur de la fenêtre de produit d'ondelettes, en nombre de points.

    for f in range(fn): # pour chaque fréquence

        v = func(f/fres,len(vals)//(2*sfreq)) # la fonction ondelette centrée
        v = v[len(v)//2 - d :len(v)//2 + d] # coupe le tableau symétriquement
        #autour de du milieu, en enleant d de chaque côté.


        for t in range(d, len(sig)-d): #calcul des coefficients P(f,t) du tableau d'ondelettes.
            #print(len(v))
            #print(len(sig[t-int(dur*sfreq/2):t+int(dur*sfreq/2)]))

            s = np.dot(sig[t - d:t + d], v) # amplitude complexe en t,f
            tab[t][f] = np.real(s)**2 + np.imag(s)**2
    return tab[d:len(sig)-d][:]



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

def maxx1(sig, fres, tab):
    '''renvoie un tableau contenant la fréquence prépondérante parmi toutes les
    fréquences, pour chaque instant'''
    maxx1 = [0 for i in range(len(tab))]
    for t in range(len(tab)):
        maxx1[t] = np.argmax(tab[t])/fres
    return maxx1

def rapp1(sig, fres, tab, maxx, maxa):
    ''' plus utilisé actuellement. rapport de Palpha sur P(fmax)'''
    rapp1 = [0 for i in range(len(tab))]
    for t in range(len(tab)):
        rapp1[t] = tab[t][int(maxa[t]*fres)]/tab[t][int(maxx[t]*fres)]
    return rapp1



def trait(raw,start, end, nb, dur, dureefenetre, channel, infa, supa):
    '''raw: l'eeg raw préchargé
    start, end: instants bornes du traitement
    nb : nombre de points dans le tableau pvar. cad nombre de fenetres
    glissantes que l'on calcule
    dur : durée sur laquelle on fait le produit d'ondelettes de raw avec la
    fonction ondelette
    dureefenetre : taille de la fenetre glissante, en secondes. Elle contient
    donc sfreq*dureefenetre points
    channel : liste des channels sur lesquels on calcule.
    infa, supa : les bornes de la bande de fréquences. '''


    pick_chans= channel #Choix du channel
    specific_chans = raw.copy().pick_channels(pick_chans)

    sig=specific_chans.filter(flo, fhi)  #filtrage du signal entre flo et fhi

    #sfreq=sig.info['sfreq'] # fréquence d'échantillonnage, ici 256Hz donnée
    # en variable globale avant.
    a =  int(sfreq * (start - dur/2))
    # pour avoir un tableau résultant entre start et end, il faut regarder
    # le signal entre start-dur/2 et end+dur/2 à cause de l'ondelette
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

    func = lambda f,t : 10000000*(np.exp(1j*2*np.pi*(f)*(vals-t)) - np.exp(-0.5*(2*np.pi*f)**2)) * np.exp(-0.34*((vals-t)**2)*(f+1)**2) #ondelette de base


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

    return [pvara, alphasurtheta];

def visualiser(start,end,channel=['EEG O1'],infa=6,supa=13):
    '''trace le pvar alpha et le rapport Palpha sur Ptheta dans
    3 figures différentes '''
    nb=2*(end-start) # on choisit de prendre 2 points par seconde

    t = np.linspace(start,end,nb)
    fig2=plt.figure()

    traitement=trapide(start,end,channel,infa,supa)

    plt.plot(t,traitement[0],'r',label="pvar alpha")
    plt.legend(loc='upper left')
    plt.show()
    fig4=plt.figure()
    plt.plot(t,traitement[1],'g',label="Palpha sur Ptheta")
    plt.legend(loc='upper left')
    plt.show()
    return;

def trapide(start,end,channel,infa,supa):
    '''traite entre start et end avec les paramètres optimaux
    (mise à jour 10/03/2018):
    - fenetre glissante de 2secondes
    - 2 points par seconde donc gros recoupement. '''
    nombrepointsparsec = 2 # un entier
    taillefenetreglissante = 2

    return trait(raw,start,end,nombrepointsparsec*int(end-start),1,taillefenetreglissante,channel,infa,supa)


def traitementadaptatif(start,end,channel=['EEG O1'],infa=6, supa=13):
    ''' renvoie un tableau de 3 tableaux, chacun longueur 2*(end-start):
        [0]: le pvar alpha
        [1]: le rapport des puissances
        [2]: les booléens result '''

    traitement= trapide(start,end,channel,infa,supa)
    return traitement

# MAIN, ici on crée les fichiers res qui contiennent le preprocessing des eegs


liste=['eeg023'] # pour tester


liste2=['eeg001','eeg002','eeg003','eeg004','eeg005','eeg006','eeg007','eeg008','eeg009',\
       'eeg010','eeg011','eeg012','eeg013','eeg014','eeg015','eeg016','eeg017',\
       'eeg018','eeg019','eeg020','eeg021','eeg021','eeg022','eeg023']


for eeg in liste2 : # pour chaque eeg

    filename= eeg
    raw = mne.io.read_raw_edf("eegs/"filename+".edf", preload=True) #lecturde de l'edf
    flo = 5 #fréq de coupure basse
    fhi = 30 #fréq de coupure haute
    sig=raw.filter(flo, fhi)

    start= 2 # normalement toujours 2 le début
    end= int(len(raw.times)/sfreq)-3 # petite marge pr être sur que ca marche

    chanlist= ['EEG O1'] # rajouter les autres channels après si besoin

    res=[]
    for chan in chanlist: # pour chaque channel
        chan=[chan]
        res.append(traitementadaptatif(start,end,chan,6,13))
    res.append(start)
    res.append(end)
    '''
    le format de res: tableau de 3 à 5 tableaux selon le nb de channels
    res[0]: un tableau des résultats de O1,
            res[0][0]: le pvar alpha
            res[0][1]: le rapport
    res[1], [2] idem si on ajoute les autres channels occipitaux
    res[3] = start
    res[4] = end '''

    newfilename='eegs/annot/res/res_'+filename+'.txt'
    file = open(newfilename, "w")

    with open(newfilename, 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(res) # sauve l'objet res dans un fichier pickle.
