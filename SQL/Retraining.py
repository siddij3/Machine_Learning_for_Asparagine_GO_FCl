
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
import shutil
import zipfile 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from enum import unique
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from  Data_to_CSV_Integrals_imports import transform_data

import math
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
from pandas import DataFrame
from multiprocessing import Process
from multiprocessing import Manager

import aws_s3
import file_management
import dicts_functions



import tensorflow as tf
import numpy as np


def Pearson(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_
    )

    tmp_numerator, tmp_denominator_real,  tmp_denominator_pred = 0, 0,0

    i = 0
    while i < len(y_pred):
        tmp_numerator += (y_true[i] - sum(y_true)/len(y_true))* (y_pred[i] - sum(y_pred)/len(y_pred))

        tmp_denominator_real += (y_true[i] - sum(y_true)/len(y_true))**2
        tmp_denominator_pred += (y_pred[i] - sum(y_pred)/len(y_pred))**2
        i += 1

    R = tmp_numerator / (math.sqrt(tmp_denominator_pred) * math.sqrt(tmp_denominator_real))

    return R[0], y_pred.flatten()

def MAE(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_
    )

    MAE = mean_absolute_error(y_true, y_pred)
    return MAE

def scaleDataset(scaleData):
    scaleData = std_scaler.fit_transform(scaleData.to_numpy())
    return DataFrame(dicts_functions.get_dict(scaleData))

def smooth_curve(points, factor=0.7):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

if __name__ == '__main__':
    
    filepath = file_management.get_data_path()
    local_data_path = os.path.expanduser(filepath)
    filenames=[]
    for filename in os.listdir(local_data_path):
        if filename.endswith('csv') and "ACE" in filename and 'Entries' in filename:
            filenames.append(filepath + "\\" + filename)

    _sum = 0

    for i in range(len(filenames)):
        df = transform_data(filenames[i])
        _sum+=len(df)

    if len(filenames)>1:
        for i in filenames[1:]:
            #print('File appended: '+i)
            df = df.append(transform_data(i),ignore_index=True,sort=False)

    dataset = shuffle(df)
    std_scaler = StandardScaler()

    # ## NEURAL NETWORK PARAMETERS
    # 
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, = file_management.importData(dataset.copy(), std_scaler)
    k_folds = 4
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 8, 8
    sum_nodes = 16 #32

    num_epochs = 10 #400 #500
    batch_size = 16 #50
    verbose = 0

    # GET FILE FROM S3 BUCKET, UNZIP IT, 

    # s3 = aws_s3.s3_bucket()
    # for bucket in s3.buckets.all():
    #         if ("wqm" in bucket.name):
    #             print(bucket.name)

    #             for obj in bucket.objects.all():
    #                 print(obj.key)

    #                 with open(obj.key, 'wb') as data:
    #                     if ("WQM" in obj.key):
    #                         s3.meta.client.download_fileobj(bucket.name, obj.key, data)
    #                         shutil.unpack_archive(obj.key)

    path = file_management.get_file_path()

    local_download_path = os.path.expanduser(path)
    optimal_NNs = [None]*k_folds

    i = 0
    for filename in os.listdir(local_download_path):
        
        if "Model" in filename:
            print(filename)
            
            optimal_NNs[i] = load_model(f"{path}\\{filename}")
            print(optimal_NNs[i].optimizer)
            i+=1

    best_architecture = [10, 8]
   
    k_fold_mae, k_models, k_weights, k_mae_history, R_tmp, history = [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds


    for fold in range(k_folds):

        reconstructed_model = optimal_NNs[fold]

        val_data = test_features[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = test_labels[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate([test_features[:i * num_val_samples], test_features[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([test_labels[:i * num_val_samples], test_labels[(i + 1) * num_val_samples:]],     axis=0)

        # reconstructed_model.compile(loss='mse', optimizer=RMSprop(0.001), metrics=['mae','mse'], run_eagerly=True)

        history = reconstructed_model.fit(
        partial_train_data, partial_train_targets,
        epochs=num_epochs, batch_size=batch_size, validation_split=0.3, verbose=verbose #, callbacks=early_stop
        )
    
        history = DataFrame(history.history)

        # ___loss, test_mae, ____mse = model.evaluate(val_data, val_targets, verbose=verbose)

        k_mae_history[fold] = history['val_mae']
        tmp = reconstructed_model.predict(partial_train_data, batch_size=None, verbose=verbose)
        print(tmp)                     
        # R_tmp[fold], y = Pearson(reconstructed_model, val_data, val_targets.to_numpy(), batch_size, verbose )
        

    best_history_retrained = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]


    dict_epochs = { 
 
        "Retrained Results": best_history_retrained,
        "Retrained Smoothed": smooth_curve(best_history_retrained)
   
    }
    dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })

    dict_epochs.to_csv(f'Evolution and Architecture Retrained - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv')

