#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.utils import class_weight


# In[2]:


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


# In[3]:


# In[4]:


def load_data():
    np.random.seed(seed)
    print('load normal')
    path = '/home/public/AI_grandchallenge_2020_4th/workspace/syj/'+str(save_name)+'/'+'normal/normal.npz'
    np_load = np.load(path)
    x_0 = np_load['nor']
    y_0 = np.zeros((len(x_0),1))
    y_0 = np_utils.to_categorical(y_0, num_classify)
    x_0 = np.array(x_0)
    y_0 = np.array(y_0)

    print(x_0.shape)
    print(y_0.shape)
    print('load violence')
    path = '/home/public/AI_grandchallenge_2020_4th/workspace/syj/'+str(save_name)+'/'+'violence/violence.npz'
    np_load = np.load(path)
    x_1 = np_load['vi']
    x_1 = np.array(x_1)
    y_1 = np.ones((len(x_1),1))
    y_1 = np_utils.to_categorical(y_1, num_classify)

    print(x_1.shape)
    print(y_1.shape)
    x_0_l, y_0_l, x_1_l, y_1_l, s_n_0, s_n_1 = [], [] ,[], [], [], []
    
    print('x_0 shape : ' + str(x_0.shape))
    print('y_0 shape : ' + str(y_0.shape))
    print('x_1 shape : ' + str(x_1.shape))

    n_0 = int(x_0.shape[0]*0.15)
    n_1 = int(x_1.shape[0]*0.15)
    
    train_x = np.concatenate([x_0[n_0:-n_0],x_1[n_1:-n_1]],axis=0)
    train_y = np.concatenate([y_0[n_0:-n_0],y_1[n_1:-n_1]],axis=0)
    
    valid_x = np.concatenate([x_0[-n_0:],x_1[-n_1:]],axis=0)
    valid_y = np.concatenate([y_0[-n_0:],y_1[-n_1:]],axis=0)
    
    test_x = np.concatenate([x_0[:n_0],x_1[:n_1]],axis=0)
    test_y = np.concatenate([y_0[:n_0],y_1[:n_1]],axis=0)
    
    
    print('train_x shape : ' + str(train_x.shape))
    print('train_y shape : ' + str(train_y.shape))
    print('valid_x shape : ' + str(valid_x.shape))
    print('valid_y shape : ' + str(valid_y.shape))
    print('test_x shape : ' + str(test_x.shape))
    print('test_y shape : ' + str(test_y.shape))
#     shuffle_l = random.sample(np.arange(len(train_x)).tolist(),len(train_x))
#     train_x = train_x[shuffle_l]
#     train_y = train_y[shuffle_l]
#     y_train = np.argmax(train_y,axis=0)
#     class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y_train),
#                                                  y_train)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def build_train():
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        inputs = Input(shape=(4001,20))
        layer = (LSTM(l_hidden, return_sequences=False))(inputs)
        layer = (Dense(d_hidden,activation='relu'))(layer)
        layer = (Dense(num_classify, activation='softmax', name='two_output'))(layer)
        model = Model(inputs, layer)
        model.summary()
        checkPoint = ModelCheckpoint(weight, monitor='val_loss', verbose=0, save_best_only=True)
        earlystop = EarlyStopping(monitor='val_loss', verbose=0, patience=patience_v)
        adam = Adam(lr=lr_v)

        model.compile(optimizer=adam,  loss='categorical_crossentropy',metrics=['accuracy'])
    hist = model.fit(train_x, train_y, batch_size=batch_size_v, epochs=epoch_v, validation_data=(valid_x,valid_y), verbose=1, callbacks=[checkPoint, earlystop]
                    )
    
    model.save(weight)
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    plt.savefig(save_fig)
    return model


# In[5]:


def test(model,test_x, test_y,threshold):
    model = load_model(weight)
    tmp_y_pred = model.predict(test_x,batch_size=batch_size_v)
    pred = copy.copy(tmp_y_pred)
    y_pred = np.argmax(tmp_y_pred, axis=1)
    y_pred_max = np.max(tmp_y_pred, axis=1)
    print('test_y' ,test_y.shape)
    test_y = np.argmax(test_y, axis=1)
    print('y_pred shape : ' + str(y_pred.shape))
    print('test_y shape : ' + str(test_y.shape))
    pred_list, test_list = [], []
    for i in range(len(y_pred)):
        pred_list.append(y_pred[i])
        test_list.append(test_y[i])
    pred_list = np.array(pred_list)
    test_list = np.array(test_list)
    print('pred_list shape : ' + str(pred_list.shape))    
    print('test_list shape : ' + str(test_list.shape))    
    
    non_idx = [test_list == 0]
    non_pred_list = pred_list[non_idx]
    non_test_list = test_list[non_idx]
    non_max_list = y_pred_max[non_idx]
    thr_idx = [non_max_list>threshold]
    thr_pred_list = non_pred_list[thr_idx]
    thr_test_list = non_test_list[thr_idx]
    cnt = 0
    for i in range(len(thr_pred_list)):
        if thr_pred_list[i]==thr_test_list[i]:
            cnt+=1
    print('accuracy : ',float(cnt/len(thr_pred_list)))
    print('loss sample : ',len(thr_pred_list)/len(non_test_list))
    print('loss sample : ',len(non_test_list), len(thr_pred_list))
    f = open(root_path+str(now_time)+'_frame_result.txt', mode='w')
    for i in range(len(test_list)):
        f.write(str(test_list[i]) + " " + str(pred_list[i]) + "\n")
    f.close()
    class_report = classification_report(test_list, pred_list,target_names = target_names)
    print(class_report)
    f = open(root_path+str(now_time)+'_frame_report.txt', mode='w')
    f.write(class_report)
    f.close()    
    cm = confusion_matrix(test_list, pred_list,normalize='true')
    print(cm)
    return tmp_y_pred


# In[6]:


def scoring_violence(pred):# only binary, raw output , 5 * 8
    tmp = np.argmax(pred,axis=1)
    cnt = Counter(tmp) 
    if len(cnt) > 1:
        if cnt.most_common(2)[0][1] == cnt.most_common(2)[1][1] :
            s_val = np.mean(pred,axis=0)
            tar = int(np.argmax(s_val))
        else:
            tar = cnt.most_common(1)[0][0]
    else:
        tar = cnt.most_common(1)[0][0]
    s_val = np.mean(pred,axis=0)
    score = s_val[int(tar)]
    
    argmax = np.argmax(pred,axis=1)
    argmax_confidence = np.max(pred,axis=1)
    return tar, score, s_val, argmax, argmax_confidence


# In[7]:


num_classify = 2
now_time = '25_wav_unit_wav_power'
root_path = '/home/public/AI_grandchallenge_2020_4th/workspace/syj/'+str(now_time)+'/'
Path(root_path).mkdir(parents=True, exist_ok=True)
save_name = now_time
weight = root_path + str(save_name)+'_tf2_last.hdf5'
save_fig = root_path + str(save_name)+'.png'
train_x, train_y, valid_x, valid_y, test_x, test_y = load_data()

print(weight)


# In[8]:


np.set_printoptions(precision=3)
batch_size_v = 512
l_hidden = 256
d_hidden = 256
lr_v = 0.001
epoch_v = 30
patience_v = 20
target_names = ['non violence','violence']
model = build_train()


# In[9]:


tmp_y_pred = test(model,test_x, test_y,0.5)

