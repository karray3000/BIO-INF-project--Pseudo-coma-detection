import os
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample

from mne import create_info, EpochsArray
from mne.baseline import rescale
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)


''' Fichier pour ouvrir les tracés eeg séparément et charger le raw .'''

raw = mne.io.read_raw_edf("eegs/eeg001.edf", preload=True ) #lecturde de l'edf


flo = 5 #fréq de coupure basse (mettre 5 ou plus pour de beaux résultats, 3-4 à vos risques et périls)
fhi = 30 #fréq de coupure haute


#pour visualiser les données raw/avoir des infos
'''
print(raw)
print(raw.info)
print(raw.ch_names)
start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
data, times = raw[:, start:stop]
print(data.shape)
print(times.shape)
data, times = raw[2:20:3, start:stop]  # access underlying data
raw.plot()
print(raw.info)

print('sample rate:', raw.info['sfreq'], 'Hz')
print('time rate',len(raw.times),'secondes')
print('%s channels x %s samples' % (len(raw), len(raw.times)))'''


"""events = mne.find_events(raw, stim_channel='SLI',verbose=False) """


#pick_chans= ['EEG O2'] #Choix du channel O1
#specific_chans = raw.pick_channels(pick_chans)

''' visualiser juste ce channel
print( specific_chans)
specific_chans.plot()
print(specific_chans.info)
'''

sig=raw.filter(flo, fhi)  #filtrage du signal entre flo et fhi

'''plt.axvline(x=np.pi,color='red')'''
raw.plot()
