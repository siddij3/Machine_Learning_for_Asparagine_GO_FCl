
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from enum import unique
from pandas import read_csv
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from keras.models import load_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error

import math
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
from pandas import DataFrame
from multiprocessing import Process
from multiprocessing import Manager


import tensorflow as tf
import numpy as np


def get_dict(tt_feats):
    dict = {
    'Time':tt_feats[:, 0], 
    'Current':tt_feats[:, 1], 
    'Spin Coating':tt_feats[:, 2] ,
    'Increaing PPM':tt_feats[:, 3], 
    'Temperature':tt_feats[:, 4], 
    'Repeat Sensor Use':tt_feats[:, 5] ,
    'Days Elapsed':tt_feats[:, 6],
    'A':tt_feats[:, 7],
    'B':tt_feats[:, 8],
    'C':tt_feats[:, 9],
    'Integrals':tt_feats[:, 10]
    }
    return DataFrame(dict)

def importData(data, scaler):

    train_dataset = data.sample(frac=0.8, random_state=5096)
    test_dataset = data.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Concentration')
    test_labels = test_features.pop('Concentration')

    train_features = get_dict(scaler.fit_transform(train_features.to_numpy()))
    test_features = get_dict(scaler.fit_transform(test_features.to_numpy()))

    #For later use
    data_labels = data.pop('Concentration')

    return data, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, 

# # Neural Network Creation and Selection Process
# 


    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)

    model = build_model(n1, n2) #, early_stop = build_model(n1, n2)

    print('Training fold #', i)
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, validation_split=0.3, verbose=verbose #, callbacks=early_stop
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    return_dict[i] = (model.to_json(), model.get_weights(), history['val_mae'], test_mae, test_R)

def KCrossLoad(i, model, features, labels, num_val_samples, epochs, batch, verbose):

    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)
    print("this was run")
    
    optimizer = RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])

    print('Training fold #', i)

    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, validation_split=0.3, verbose=verbose #, callbacks=early_stop
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    return model.to_json(), model.get_weights(), history['val_mae'], test_mae, test_R

def Pearson(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_,
        workers=3,
        use_multiprocessing=True,
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
        verbose=verbose_,
        workers=3,
        use_multiprocessing=True,
    )

    MAE = mean_absolute_error(y_true, y_pred)
    return MAE

def scaleDataset(scaleData):
    scaleData = std_scaler.fit_transform(scaleData.to_numpy())
    return DataFrame(get_dict(scaleData))

def smooth_curve(points, factor=0.7):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# ## Functions for Isolating Parameters
def loop_folds(neuralNets, _predictions, R, mae, k_folds, features, labels,   param1, param2, inner_val, outer_val, batch, vbs, str_test):

    tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds
    avg_predictions = [None]*k_folds

    for j, NN in enumerate(neuralNets):
        test_mae = MAE(NN, features, labels, batch, vbs)

        tmp, tmp_predictions = Pearson(NN, features, labels, batch, vbs) 
        tmp_R[j] = tmp
        tmp_mae[j] = test_mae

        #dict_title = f"Predicted NN {j} for {param1},{inner_val}; {param2},{outer_val} - {str_test}"
        #_predictions[dict_title] = tmp_predictions.tolist()

        avg_predictions[j] = tmp_predictions.tolist()
    
    dict_average = f"Averages for {param1},{inner_val}; {param2},{outer_val}"
    dict_title_real = f"Real for {param1},{inner_val}; {param2},{outer_val} - {str_test}"

    _predictions[dict_title_real] = labels.tolist()

    arr_avg_predictions = np.transpose(avg_predictions)
    _predictions[dict_average] = [np.mean(i) for i in arr_avg_predictions]

    R.append(sum(tmp_R)/len(tmp_R))
    mae.append(sum(tmp_mae)/len(tmp_mae))

    return _predictions, R, mae

def isolateParam(optimal_NNs, data, parameter, batch, verbose, str_test): 
    # Split the data labels with time

    unique_vals = np.unique(data[parameter]) #Repeat Sensor Use
    param_index= [np.where(data[parameter].to_numpy()  == i)[0] for i in unique_vals]

    scaled_features = scaleDataset(all_features.copy())
    #The full features of the data points that use certain time values

    param_features =  [scaled_features.iloc[param_index[int(i)]] for i in unique_vals]


    param_labels = [data_labels.to_numpy()[param_index[int(i)]] for i in unique_vals]

    mae, R = [], []
    _predictions = {}

    for i in unique_vals:
        print(f'{parameter}: {i}')

        _predictions, R, mae = loop_folds(optimal_NNs, _predictions, 
        R, mae, 
        k_folds, 
        param_features[int(i)], param_labels[int(i)],   
        parameter, "", 
        int(i), None, 
        batch, verbose, str_test)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{str_test} - {parameter} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)
    
    average_R = [i for i in R]
    average_mae = [i for i in mae]

    return average_R, average_mae
 
def isolateTwoParam(optimal_NNs, data, parameter1, parameter2, batch, vbs, str_test):

    unique_vals_inner = np.unique(data[parameter1]) #Repeat Sensor Use
    unique_vals_time = np.unique(data[parameter2]) #Times

    inner = [[x for _, x in data.groupby(data[parameter1] == j)  ][1] for j in unique_vals_inner]   
    time_use = [[[x.index.values for _, x in data.groupby(val[parameter2] == j)  ][1] for val in inner] for j in unique_vals_time] 
 
    scaled_features = scaleDataset(all_features.copy())
    
    feats = [[scaled_features.iloc[sc]  for sc in rsu] for rsu in time_use] 
    labels = [[data_labels.to_numpy()[sc]  for sc in rsu] for rsu in time_use]

    tr_mae = []
    tr_R = []
    _predictions = {}
    for i, time_vals in enumerate(feats):
        tr_tmp_mae, tr_tmp_R = [], []

        for j, rsu_vals in enumerate(time_vals):

            print(f'{parameter1}: {unique_vals_inner[j]}', f'{parameter2}: {unique_vals_time[i]}')
            
            _predictions, tr_tmp_R, tr_tmp_mae = loop_folds(optimal_NNs, _predictions, 
            tr_tmp_R, tr_tmp_mae, 
            k_folds, 
            rsu_vals, labels[i][j],   
            parameter1, parameter2, 
            j, i, 
            batch, vbs, str_test)


        tr_mae.append(tr_tmp_mae)
        tr_R.append(tr_tmp_R)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{str_test} {parameter1} and {parameter2} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)

 
    averages_mae = [[j for j in i] for i in tr_mae] 
    averages_R = [[j for j in i] for i in tr_R] 

    return averages_R, averages_mae

def daysElapsed(optimal_NNs, data, param1, param2,  batch, vbs): 
    # Split the data labels with spin coating first

    param3 = 'Time'
    unique_vals_sc = np.unique(data[param1]) #Spin Coating
    unique_vals_days = np.unique(data[param2]) #Days Elapsed
    unique_vals_time = np.unique(data[param3])

    days = [[x for _, x in data.groupby(data[param2] == j)  ][1] for j in unique_vals_days]
    time_days = [[[x for _, x in data.groupby(val[param3] == j)  ][1] for val in days] for j in unique_vals_time] 

    #[time][day elapsed][spin coated]    
    all_vals = [[[x.index.values for _, x in data.groupby(unique_days[i][param1] == 0)] for i,val in enumerate(unique_days)] for unique_days in time_days] 
    
    scaled_features = scaleDataset(all_features.copy())

    feats = [[[scaled_features.iloc[sc]  for sc in days] for days in times] for times in all_vals]
    labels = [[[data_labels.to_numpy()[sc]  for sc in days] for days in times] for times in all_vals]

    shared_mae, shared_R = [], []
    _predictions = {}


    for t, times in enumerate(feats):
        days_tmp_mae, days_tmp_R = [], []

        for d, days in enumerate(times):
            sc_tmp_mae, sc_tmp_R = [], []

            print(f'{param2}: {unique_vals_days[d]}', f'{param3}: {unique_vals_time[t]}')

            for sc, isSpin in enumerate(days):

                _predictions, sc_tmp_R, sc_tmp_mae = loop_folds(optimal_NNs, _predictions, 
                sc_tmp_R, sc_tmp_mae, 
                k_folds, 
                isSpin, labels[t][d][sc],   
                param1, param2, 
                d, sc, 
                batch, vbs, "All data")

            days_tmp_mae.append(sc_tmp_mae)
            days_tmp_R.append(sc_tmp_R)

        shared_mae.append(days_tmp_mae)
        shared_R.append(days_tmp_R)

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'{param1} - {param2} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)

    R_ = {}
    MAE_  = {}

    for d, days in enumerate(shared_R[0]):
        for sc, isSC in enumerate(shared_R[0][d]):
            tmp_time_R, tmp_time_MAE = [], []

            tmp_time_R = [ time[d][sc] for t, time in enumerate(shared_R)]
            tmp_time_MAE = [ shared_mae[t][d][sc]  for t, time in enumerate(shared_R)]

            MAE_title = f" {param2} {unique_vals_days[d]}: {param1} {sc}; MAE"
            R_title = f"   {param2} {unique_vals_days[d]}: {param1} {sc}; R"
            
            MAE_[MAE_title] = tmp_time_MAE
            R_[R_title] = tmp_time_R

    R_MAE = R_ | MAE_
    #MAE_ = DataFrame({ key:pd.Series(value) for key, value in R_MAE.items() })

    return R_MAE

def create_dict(str_param, loop_values, R_val, mae_val, R_val_test, mae_val_test):

    str_r =  'R {} '.format(str_param)
    str_mae =  'MAE {}'.format(str_param)    
    
    str_r_test =  'R Test {} '.format(str_param)
    str_mae_test =  'MAE Test {}'.format(str_param)

    return {
    str_param: [i for i in loop_values],
    str_r    : R_val , 
    str_mae  : mae_val,

    str_r_test: R_val_test,
    str_mae_test: mae_val_test
    }

def create_dict_two(str_param, loop_values, R_val, mae_val):

    str_r =  'R {} '.format(str_param)
    str_mae =  'MAE {}'.format(str_param)


    return {
    str_param: [i for i in loop_values],
    str_r    : R_val , 
    str_mae  : mae_val 
    }

if __name__ == '__main__':
    
    dataset = shuffle(read_csv('aggregated_integrals.csv'))
    std_scaler = StandardScaler()

    # ## NEURAL NETWORK PARAMETERS
    # 
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, = importData(dataset.copy(), std_scaler)
    k_folds = 4
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 8, 8
    sum_nodes = 16 #32

    num_epochs = 10 #400 #500
    batch_size = 16 #50
    verbose = 0

    # 
    path = ".\\"
    local_download_path = os.path.expanduser(path)
    
    optimal_NNs = [None]*k_folds
    i = 0
    for filename in os.listdir(local_download_path):
        if "Model" in filename and 'number' in filename:
            print(filename)
            
            optimal_NNs[i] = load_model(f'{filename}')
            i+=1

    best_architecture = [10, 8]
    # optimal_NNs = [load_model(f'{path}Model {best_architecture} number 0'), load_model(f'{path}Model {best_architecture} number 1'),  load_model(f'{path}Model {best_architecture} number 2'), load_model(f'{path}Model {best_architecture} number 3') ]
   
    k_fold_mae, k_models, k_weights, k_mae_history, R_tmp, history = [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds

    #model_json, model_weights, k_mae_history, k_fold_mae, R_tmp =  KCrossLoad(0, optimal_NNs[0], test_features, test_labels, num_val_samples, num_epochs, batch_size, verbose)

    for fold in range(k_folds):

        model = optimal_NNs[fold]
        print(model.optimizer)

        val_data = test_features[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = test_labels[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate([test_features[:i * num_val_samples], test_features[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([test_labels[:i * num_val_samples], test_labels[(i + 1) * num_val_samples:]],     axis=0)
        print("this was run")
        
        optimizer = RMSprop(0.001)
        # model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])

        print('Training fold #', fold)

        history = model.fit(
            partial_train_data, partial_train_targets,
            epochs=num_epochs, batch_size=batch_size, validation_split=0.3, verbose=verbose #, callbacks=early_stop
        )

        history = DataFrame(history.history)

        # ___loss, test_mae, ____mse = model.evaluate(val_data, val_targets, verbose=verbose)


        k_mae_history[fold] = history['val_mae']
        print(k_mae_history)
        # R_tmp[fold], y = Pearson(model, val_data, val_targets.to_numpy(), batch_size, verbose )


    best_history_retrained = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]


    dict_epochs = { 
 

        "Retrained Results": best_history_retrained,
        "Retrained Smoothed": smooth_curve(best_history_retrained)
   
    }
    dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })

    dict_epochs.to_csv('Evolution and Architecture Retrained - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)

