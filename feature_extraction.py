#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math 
import matplotlib.pyplot as plt
import os
import wave
import scipy
import numpy as np
import librosa
import h5py
import sys
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, SimpleRNN, LSTM,Permute, Bidirectional, RepeatVector, Masking
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, ThresholdedReLU

from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import History, ModelCheckpoint,EarlyStopping
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from confusion import plot_confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import seaborn as sn
import pandas as pd
import random
import copy
seed = 7
np.random.seed(seed)
from tqdm.auto import tqdm
from datetime import datetime
from pytz import timezone, utc
KST = datetime.now(timezone('Asia/Seoul'))
fmt = "%Y_%m_%d_%H_%M_%S"
now_time = KST.strftime(fmt)
print(now_time)
physical_devices_list = tf.config.list_physical_devices('GPU') 
physical_devices = physical_devices_list[:4]
tf.config.set_visible_devices(physical_devices, 'GPU')
print(physical_devices)
tf.config.set_soft_device_placement(True)
for idx, device in enumerate(physical_devices):
    tf.config.experimental.set_memory_growth(device,True)
from jupyterthemes import jtplot
jtplot.style(theme='grade3')
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logging.disable(sys.maxsize)
from cal_score import scoring
from pathlib import Path
from collections import Counter 
import seaborn as sns
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
parser = argparse.ArgumentParser(description='python Implementation')
parser.add_argument('--input_dir', type = str, default =None, help='input_dir')
parser.add_argument('--output_dir', type = str, default =None, help='input_dir')
args = parser.parse_args()
# In[33]:


path = args.input_dir
output_path = args.output_dir
json_path = path+'/wav_info/'
wav_path = path+'/wav_clip_16k_36_all/'
vio_path = output_path+'/violence/'
nor_path = output_path+'/normal/'
sil_path = output_path+'/silence/'
json_files = os.listdir(json_path)
Path(vio_path).mkdir(parents=True, exist_ok=True)
Path(nor_path).mkdir(parents=True, exist_ok=True)
Path(sil_path).mkdir(parents=True, exist_ok=True)
vi, nor, sil = [], [], []
for one_file in tqdm(json_files):
    pivot_l = []
    
    with open(json_path+one_file) as json_file:
        y, sr = librosa.load(wav_path+one_file.replace('.json','.wav'), sr=16000)
        total_time = len(y)/sr
        json_data = json.load(json_file)
        try:
            audios = json_data['data']['scene']['audio']
        except:
            audios = json_data['data']['scene_000']['audio']
        audios_keys = audios.keys()
#         print(audios_keys)
        splits = one_file.split('.')[0].split('_')
        clip = splits[0]
        segment = splits[1]
        y_spec = librosa.feature.mfcc(y, n_mfcc=20, n_fft = 400, hop_length=160)
        y_spec = np.transpose(y_spec,(1,0))
#         y_spec = min_max(y_spec,max_l,min_l)
#         y_spec = scaler.fit_transform(y_spec)
        
        for one_key in audios_keys:
#             print(audios[one_key])
            start = audios[one_key]['onset'] * sr
            end = audios[one_key]['offset'] * sr
            val = audios[one_key]['valence']
            arousal = audios[one_key]['arousal']
            violence = audios[one_key]['violence']
            emotion = audios[one_key]['emotion']
            text = audios[one_key]['text']
            pivot_l.append([start,end])
            sig_len = len(y)
            y2 = y_spec[round((start/sig_len)*y_spec.shape[0]):round((end/sig_len)*y_spec.shape[0])]
            y_remember = y[round((start/sig_len)*y.shape[0]):round((end/sig_len)*y.shape[0])]
            if y2.shape[0]<32 or round((start/sig_len)*y_spec.shape[0])== round((end/sig_len)*y_spec.shape[0]):
                continue
            if violence == 'violence':
                vi.append(y2)
                librosa.output.write_wav(vio_path+one_file.replace('.json','')+'_'+one_key+'_'+str(int(val)).zfill(2)+'_'+str(int(arousal)).zfill(2)+'_'+violence[:2]+'_'+emotion[:2]+'.wav', y_remember, sr)
            elif violence =='normal':
                nor.append(y2)
                librosa.output.write_wav(nor_path+one_file.replace('.json','')+'_'+one_key+'_'+str(int(val)).zfill(2)+'_'+str(int(arousal)).zfill(2)+'_'+violence[:2]+'_'+emotion[:2]+'.wav', y_remember, sr)

        pivot_l = np.array(pivot_l)
        if pivot_l.shape[0] == 0:
            continue
        pivot_sorted = np.sort(pivot_l,axis=0)
        full = np.r_[[[0,0]],pivot_sorted,[[sig_len,sig_len]]]
        for i in range(len(full)-1):
            start = full[i][1]
            end = full[i+1][0]
            y2 = y_spec[int(round((start/sig_len)*y_spec.shape[0])):int(round((end/sig_len)*y_spec.shape[0]))]
            if y2.shape[0]<32 or round((start/sig_len)*y_spec.shape[0])== round((end/sig_len)*y_spec.shape[0]):
                    continue
            sil.append(y2)
            librosa.output.write_wav(sil_path+one_file.replace('.json','')+'_'+one_key+'_'+str(int(val)).zfill(2)+'_'+str(int(arousal)).zfill(2)+'_'+violence[:2]+'_'+emotion[:2]+'.wav', y_remember, sr)

np.savez(vio_path+'violence.npz',vi = vi)
np.savez(nor_path+'normal.npz',nor = nor)
np.savez(sil_path+'silence.npz',sil = sil)


# In[ ]:




