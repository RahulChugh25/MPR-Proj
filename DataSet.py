#!/usr/bin/env python
# coding: utf-8

# In[637]:


import numpy as np
import keras as kr
import librosa


# In[638]:


from pydub import AudioSegment
from os import path


# In[639]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[641]:


data = pd.read_csv('datatype.csv')


# In[642]:


data.head()


# In[643]:


# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)


# In[ ]:





# In[ ]:





# In[644]:


encoder = LabelEncoder()


# In[645]:


scaler = StandardScaler()


# In[ ]:





# In[646]:


X = scaler.fit_transform(np.array(data.iloc[:,:-1],dtype = float))


# In[647]:


genre_list = data.iloc[:,-1]
y = encoder.fit_transform(genre_list)


# In[648]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[649]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))


# In[650]:


model.compile(optimizer='adam' ,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
adam = kr.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)


# In[651]:


history = model.fit(X_train,
                    y_train,
                    epochs=200,
                    batch_size=10)


# In[652]:


eva = model.predict(X_test) #To print results on 3000 dataset using test data


# In[ ]:





# In[654]:


data2 = pd.read_csv('data.csv')


# In[655]:


data2 = data2.drop(['filename'],axis=1)


# In[656]:


i = scaler.fit_transform(np.array(data2.iloc[:,:-1],dtype = float))


# In[657]:


gn = data2.iloc[:,-1]
j = encoder.fit_transform(gn)


# In[ ]:





# In[658]:


test_loss, test_acc = model.evaluate(X_test,y_test)
test_loss2, test_acc2 = model.evaluate(i,j)


# In[659]:


test_acc


# In[660]:


test_acc2


# In[ ]:





# In[674]:


header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for count in range(1, 21):
    header += f' mfcc{count}'
header += ' label'
header = header.split()

file = open('data3.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

def met(songname,g):
    y, sr = librosa.load(songname, mono=True, duration = 10)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    to_append = f'songname {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append += f' {g}'
    file = open('data3.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())


# In[675]:


met("_Mitwa(PagalWorld.com).wav",'rock')
met("03. Rock On!!.wav",'rock')
met("Teri Mitti - Songs.pk - 320Kbps.wav",'classic')
met("Zinda.wav",'pop')


# In[ ]:





# In[676]:


data3 = pd.read_csv('data3.csv')

data3 = data3.drop(['filename'],axis=1)


# In[677]:


data3.head()


# In[678]:


k = scaler.fit_transform(np.array(data3.iloc[:,:-1],dtype = float))


# In[679]:


gn2 = data3.iloc[:,-1]
l = encoder.fit_transform(gnp)


# In[ ]:





# In[680]:


m = model.predict(k) #For self created data


# In[681]:


test_loss3 , test_acc3 = model.evaluate(k,l)


# In[682]:


test_acc3


# In[683]:


for cn in m:
    print(cn.argmax())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




