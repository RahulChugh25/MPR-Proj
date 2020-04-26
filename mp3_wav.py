#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pathlib
from pydub import AudioSegment


# In[8]:


genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()


# In[21]:


for g in genres:
    for filename in os.listdir(f'./GTZAN/{g}'):
        songmp3 = f'./GTZAN/{g}/{filename}'
        songwav = filename.replace('.au','.wav')
        songwavdir = f'./GTZAN/{g}/{songwav}'
        sound = AudioSegment.from_file(songmp3,'au')
        sound.export(songwavdir, format="wav")


# In[9]:


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


# In[10]:


import numpy as np
import pandas as pd
import csv
import librosa


# In[16]:





# In[6]:


file = open('datanew2.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
for g in genres:
    for filename in os.listdir(f'./GTZAN/{g}'):
        if filename.endswith('.wav'):
            songname = f'./GTZAN/{g}/{filename}'
            off = 0
            for i in range(0,3):
                off = i * 10
                y, sr = librosa.load(songname, mono=True, offset = off, duration = 10)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                rms = librosa.feature.rms(y=y)
                to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                to_append += f' {g}'
                file = open('datanew2.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())


# In[ ]:





# In[ ]:





# In[ ]:




