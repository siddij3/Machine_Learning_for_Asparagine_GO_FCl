
# ## Importing Data


# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from pandas import read_csv
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
import math
import pandas as pd
from pandas import DataFrame


import numpy as np


dataset = read_csv('.\Data\\aggregated_data.csv')
dataset = shuffle(dataset)

std_scaler = StandardScaler()


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
def loop_folds(neuralNets, _predictions, R, mae, k_folds, features, labels, param1, param2, inner_val, outer_val, batch, vbs, str_test):

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

    param_features =  [scaled_features.iloc[param_index[int(i)]] for i, _ in enumerate(unique_vals)]


    param_labels = [data_labels.to_numpy()[param_index[int(i)]] for i, _ in enumerate(unique_vals)]
    print(len(param_labels))
    print(len(param_labels[0]))
    print(len(param_labels[1]))
    print(len(param_labels[2]))

    mae, R = [], []
    _predictions = {}

    for i, unique_val in enumerate(unique_vals):
        print(f'{parameter}: {unique_val}')

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
    time_use = [[[x.index.values for _, x in data.groupby(val[parameter2] == j)  ][-1] for val in inner] for j in unique_vals_time] 
 
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

if __name__ == '__main__':

    dataset = read_csv('.\Data\\aggregated_data_ace1.csv')
    dataset = shuffle(dataset)
    
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, = importData(dataset.copy(), std_scaler)
    k_folds = 4
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 6,6
    sum_nodes = 32 #32

    num_epochs = 400 #500
    batch_size = 32 #50
    verbose = 0

    #filepath = '.\\epochs {} - Sum {} - Epochs {} - Batch {} - Data {}\\'.format(num_epochs, sum_nodes, batch_size, "both")

    path = ".\\Test Models\\"
    nnmodel = [21,10]
    optimal_NNs = [ load_model(f'{path}Model {nnmodel} number 0'), load_model(f'{path}Model {nnmodel} number 1'),  load_model(f'{path}Model {nnmodel} number 2'), load_model(f'{path}Model {nnmodel} number 3') ]

    print("\n")
    print("path: ", path)
    print(f"Model: Model {nnmodel}")
    print("\n")
    # Scaling Data Set Function

    # 
    start_index= 0
    end_index = 3
    vbs = 0
    start_time = 1
    param_batches = 10

    str_reg = "All"
    str_test = "Test"

    str_time = 'Time'
    str_increasing = 'Increasing PPM'
    str_spin =  'Spin Coating'
    str_repeat = 'Repeat Sensor Use'
    str_days = 'Days Elapsed'

    str_a = 'A'
    str_b = 'B'
    str_c = 'C'

    print("Isolating Spin Coating and Time")
    #R_of_sc , mae_of_sc  = isolateTwoParam(optimal_NNs, all_features, str_spin, str_days, param_batches, vbs, str_reg)
    #R_of_rsu_testdata, mae_of_rsu_testdata = 
    R_of_sc , mae_of_sc  = isolateParam(optimal_NNs , all_features, str_repeat, param_batches, vbs, str_reg )

    # R_of_sct_testdata, mae_of_sct_testdata = isolateTwoParam(optimal_NNs, test_dataset, 'Spin Coating', str_increasing, param_batches, vbs, str_test)

    dict_sc = {
    "SC: R"    : R_of_sc ,
    "SC: MAE"  : mae_of_sc ,

     "Time SC: 0;: R"    : [i[0] for i in R_of_sc ], 
     "Time SC: 1: R"    : [i[1] for i in R_of_sc ],     
     "Time SC: 0 : MAE"  : [i[0] for i in mae_of_sc ], 
     "Time SC: 1 : MAE"  : [i[1] for i in mae_of_sc ],

    }
 

    # # Printing to CSV

    dict_all = dict_sc 
    dict_all = DataFrame({ key:pd.Series(value) for key, value in dict_all.items() })
    dict_all.to_csv(f'Final - Days Elapsed Only.csv'.format(sum_nodes, num_epochs, k_folds), index=False)

  
