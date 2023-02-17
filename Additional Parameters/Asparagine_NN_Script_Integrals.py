
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from enum import unique
from pandas import read_csv
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
#from keras.callbacks import EarlyStopping
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
def build_model(n1, n2):
  #Experiment with different models, thicknesses, layers, activation functions; Don't limit to only 10 nodes; Measure up to 64 nodes in 2 layers
  
    model = Sequential([
    layers.Dense(n1, activation=tf.nn.relu, input_shape=[11]),
    layers.Dense(n2, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
    #early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) 

    return model #, early_stop

def KCrossValidation(i, features, labels, num_val_samples, epochs, batch, verbose, n1, n2, return_dict):

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
    #dataset = read_csv('aggregated_data.csv')

    filepath = r".\\Data\\"
    #filepath = r"C:\\Users\\junai\\Documents\\McMaster\\Food Packaging\\Thesis\\Thesis Manuscript\\Experiments\\Raw Data\\ML Parsed"
    local_download_path = os.path.expanduser(filepath)
    filenames=[]
    for filename in os.listdir(local_download_path):
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
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, = importData(dataset.copy(), std_scaler)
    k_folds = 4
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 8, 8
    sum_nodes = 18 #32

    num_epochs = 50 #400 #500
    batch_size = 16 #50
    verbose = 0

    print("\n")
    print("Number Folds: ", k_folds)
    print("Initial Layers: ", n1_start, n2_start)
    print("Total Nodes: ", sum_nodes)
    print("Epochs: ", num_epochs)
    print("Batch Size: ", batch_size)
    print("\n")

    best_architecture = [0,0]

    dict_lowest_MAE,dict_highest_R  = {}, {}
    best_networks, best_history = 0,0

    mae_best  = 10
    R_best  = 0

    # #### Where the Magic Happens
    for i in range(n2_start, sum_nodes):
        for j in range(n1_start, sum_nodes):
            if (i+j > sum_nodes):
                continue
            
            print("first hidden layer", j)
            print("second hidden layer", i)
            k_fold_mae, k_models, k_weights, k_mae_history, R_tmp = [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds

            _futures = [None]*k_folds
            manager = Manager()
            return_dict = manager.dict()

            for fold in range(k_folds):
                _futures[fold] = Process(target=KCrossValidation, 
                                              args=(  fold, 
                                                train_features, 
                                                train_labels, 
                                                num_val_samples, 
                                                num_epochs, 
                                                batch_size, 
                                                verbose, 
                                                j, 
                                                i, return_dict))
                _futures[fold].start()   
                
            for job in _futures:
                job.join()

        # (model.to_json(), model.get_weights(), history['val_mae'], test_mae, test_mse, test_R)
            for fold in range(k_folds):
                k_models[fold] = model_from_json(return_dict.values()[fold][0]) #model is a JSON file
                k_weights[fold] = return_dict.values()[fold][1]
                k_models[fold].set_weights(k_weights[fold])

                k_mae_history[fold] = return_dict.values()[fold][2]
                k_fold_mae[fold] = return_dict.values()[fold][3]

                R_tmp[fold] = return_dict.values()[fold][4]

            

            R_recent = sum(R_tmp)/len(R_tmp)
            mae_recent = sum(k_fold_mae)/len(k_fold_mae)


            dict_highest_R['R: {}, {}'.format(j, i)] = R_recent
            dict_lowest_MAE['MAE: {}, {}'.format(j, i)] = mae_recent

            if (mae_recent <= mae_best):
                mae_best = mae_recent
                best_networks = k_models
                best_architecture = [j,i]
                best_history = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]
            
            print(mae_best, mae_recent, best_architecture)


    # 
    # Find the model with the lowest error
    optimal_NNs  = best_networks
    i = 0
    for model in optimal_NNs :
        model.save("Model {} number {}".format(best_architecture, i))
        print("Models saved")

        i +=1

    # Plotting Loss Transition
    smooth_mae_history = smooth_curve(best_history)



    dict_epochs = { 
        "Epochs" : range(1, len(best_history) + 1),
        "Lowest MAE": best_history,

        "Smoothed Epochs": range(1, len(smooth_mae_history) + 1),

        "Lowest MAE Smoothed": smooth_mae_history
   
    }

    dict_epochs = dict_epochs | dict_highest_R | dict_lowest_MAE
    dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })

    dict_epochs.to_csv('Evolution and Architecture - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)

    # 
    # start_index= 0
    # end_index = 3
    # vbs = 0
    # start_time = 1
    # param_batches = 10

    # str_reg = "All"
    # str_test = "Test"

    # str_time = 'Time'
    # str_increasing = 'Increasing PPM'
    # str_spin =  'Spin Coating'
    # str_days = 'Repeat Sensor Use'
    # str_repeat = 'Days Elapsed'

    # str_a = 'A'
    # str_b = 'B'
    # str_c = 'C'



    # #  Isolating Spin Coating
    # # print("Isolating Spin Coating")
    # R_of_sc , mae_of_sc  = isolateParam(optimal_NNs , all_features, str_spin, param_batches, vbs, str_reg )
    # R_of_sc_testdata, mae_of_sc_testdata = isolateParam(optimal_NNs, test_dataset,  str_spin, param_batches, vbs, str_test )

    # print("Isolating Spin Coating and Time")
    # R_of_sct , mae_of_sct  = isolateTwoParam(optimal_NNs, all_features, str_spin, 'Time', param_batches, vbs, str_reg)
    # R_of_sct_testdata, mae_of_sct_testdata = isolateTwoParam(optimal_NNs, test_dataset, str_spin, 'Time', param_batches, vbs, str_test)

    # dict_sc = {
    #     "SC: R"    : R_of_sc ,
    #     "SC: MAE"  : mae_of_sc ,
    #     "SC Test: R"    : R_of_sc_testdata,
    #     "SC Test: MAE"  : mae_of_sc_testdata,
        
    #     "Time SC: 0;: R"    : [i[0] for i in R_of_sct ], 
    #     "Time SC: 1: R"    : [i[1] for i in R_of_sct ],     
    #     "Time SC: 0 : MAE"  : [i[0] for i in mae_of_sct ], 
    #     "Time SC: 1 : MAE"  : [i[1] for i in mae_of_sct ], 
    #     "Time SC: 0; Test R"    : [i[0] for i in R_of_sct_testdata], 
    #     "Time SC: 1; Test R"    : [i[1] for i in R_of_sct_testdata],     
    #     "Time SC: 0; Test MAE"  : [i[0] for i in mae_of_sct_testdata], 
    #     "Time SC: 1; Test MAE"  : [i[1] for i in mae_of_sct_testdata], 
    # }
    # #  Isolating Spin Coating and Time
    # #  Isolating Time
    # print("Isolating Time")
    # R_time , mae_averages_time  = isolateParam(optimal_NNs , all_features, 'Time', param_batches, vbs, str_reg )
    # R_time_testdata, mae_averages_time_testdata = isolateParam(optimal_NNs , test_dataset, 'Time',param_batches, vbs, str_test )

    # dict_time = {
    #     "Time": [i for i in range(0, 51)],
    #     "Time:  R"    : R_time , 
    #     "Time:  MAE"  : mae_averages_time , 
    #     "Time: Test R"    : R_time_testdata, 
    #     "Time: Test MAE"  : mae_averages_time_testdata
    #     }

    # #  Isolating Increasing
    # print("Isolating Increasing")
    # R_of_increasing , mae_of_increasing  = isolateParam(optimal_NNs , all_features, 'Increasing PPM', param_batches, vbs, str_reg )
    # R_of_increasing_testdata, mae_of_increasing_testdata = isolateParam(optimal_NNs, test_dataset,  'Increasing PPM', param_batches, vbs, str_test )

    # print("Increasing PPM and Time")
    # R_of_increasing_time , mae_of_increasing_time  = isolateTwoParam(optimal_NNs, all_features, 'Increasing PPM', 'Time', param_batches, vbs,str_reg )
    # R_of_increasing_time_testdata, mae_of_increasing_time_testdata = isolateTwoParam(optimal_NNs , test_dataset, 'Increasing PPM', 'Time', param_batches, vbs, str_test)

    # dict_inc = {
    #     "Increasing:  R"    : R_of_increasing , 
    #     "Increasing:  MAE"  : mae_of_increasing , 
    #     "Increasing: Test R"    : R_of_increasing_testdata, 
    #     "Increasing: Test MAE"  : mae_of_increasing_testdata,

    #     "Time Increasing: 0 : R"    : [i[0] for i in R_of_increasing_time ], 
    #     "Time Increasing: 1 : R"    : [i[1] for i in R_of_increasing_time ],     
    #     "Time Increasing: 0 : MAE"  : [i[0] for i in mae_of_increasing_time ] , 
    #     "Time Increasing: 1 : MAE"  : [i[1] for i in mae_of_increasing_time ], 

    #     "Time Increasing: 0; Test R"    : [i[0] for i in R_of_increasing_time_testdata], 
    #     "Time Increasing: 1; Test R"    : [i[1] for i in R_of_increasing_time_testdata],     
    #     "Time Increasing: 0; Test MAE"  : [i[0] for i in mae_of_increasing_time_testdata] , 
    #     "Time Increasing: 1; Test MAE"  : [i[1] for i in mae_of_increasing_time_testdata], 
    # }

    # #  Isolating Repeat Sensor Use
    # #print("Isolating Repeat Sensor Use")
    # #R_of_rsu , mae_of_rsu  = isolateParam(optimal_NNs , all_features, 'Repeat Sensor Use', param_batches, vbs, str_reg )
    # #R_of_rsu_testdata, mae_of_rsu_testdata = isolateParam(optimal_NNs, test_dataset,  'Repeat Sensor Use', param_batches, vbs, str_test )
    # print("Isolating Repeat Sensor Use and Time")
    # R_of_tr , mae_of_tr  = isolateTwoParam(optimal_NNs, all_features, 'Repeat Sensor Use', 'Time', param_batches, vbs, str_reg)

    # dict_repeat = {

    #     "Use 1, Time: R"    : [i[0] for i in R_of_tr ], 
    #     "Use 2, Time: R"    : [i[1] for i in R_of_tr ], 
    #     "Use 3, Time: R"    : [i[2] for i in R_of_tr ], 

    #     "Use 1, Time: MAE"    : [i[0] for i in mae_of_tr ], 
    #     "Use 2, Time: MAE"    : [i[1] for i in mae_of_tr ], 
    #     "Use 3, Time: MAE"    : [i[2] for i in mae_of_tr ]
    #     }

    # #Isolating A, B, C
    # print("Isolating ABC")
    # R_of_A , mae_of_A  = isolateParam(optimal_NNs , all_features, 'A', param_batches, vbs, str_reg )
    # R_of_B , mae_of_B  = isolateParam(optimal_NNs , all_features, 'B', param_batches, vbs, str_reg )
    # R_of_C , mae_of_C  = isolateParam(optimal_NNs , all_features, 'C', param_batches, vbs, str_reg )

    # dict_abc = {
    #     "A:  R"   : R_of_A , 
    #     "B:  R"   : R_of_B , 
    #     "C:  R"   : R_of_C, 
    #     "A:  MAE" : mae_of_A , 
    #     "B:  MAE" : mae_of_B , 
    #     "C:  MAE" : mae_of_C, 
    # }

    # # Isolating Days Elapsed
    # #R_of_sc , mae_of_sc  = isolateParam(optimal_NNs , all_features, 'Days Elapsed', param_batches, vbs, str_reg )
    # #R_of_sc_testdata, mae_of_sc_testdata = isolateParam(optimal_NNs, test_dataset,  'Days Elapsed', param_batches, vbs, str_test )

    # #  Repeat Sensor Use
    # print("Days Elapsed and Spin Coating")                   
    # dict_daysElapsed_time = daysElapsed(optimal_NNs, all_features, 'Spin Coating', 'Days Elapsed',  param_batches, vbs)

    # # # Printing to CSV

    # dict_all = dict_sc | dict_time | dict_inc | dict_repeat | dict_abc | dict_daysElapsed_time
    # dict_all = DataFrame({ key:pd.Series(value) for key, value in dict_all.items() })
    # dict_all.to_csv('Final - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)

    # print(best_architecture)
