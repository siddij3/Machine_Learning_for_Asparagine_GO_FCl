
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

def KCrossValidation(i, features, labels, num_val_samples, epochs, batch, verbose, n1, n2):

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

    return model, history['val_mae'], test_mae, test_R

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

    n1_start, n2_start = 15, 15
    sum_nodes = 30 #32

    num_epochs = 500 #400 #500
    batch_size = 16 #50
    verbose = 0

    print("\n")
    print("Number Folds: ", k_folds)
    print("Initial Layers: ", n1_start, n2_start)
    print("Total Nodes: ", sum_nodes)
    print("Epochs: ", num_epochs)
    print("Batch Size: ", batch_size)
    print("\n")

    best_architecture = [n1_start, n2_start]

    dict_lowest_MAE,dict_highest_R  = {}, {}
    best_networks, best_history = 0,0

    mae_best  = 10
    R_best  = 0

    # #### Where the Magic Happens

    k_fold_mae, k_models, k_weights, k_mae_history, R_tmp = [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds


    for fold in range(k_folds):
        k_models[fold], k_mae_history[fold],  k_fold_mae[fold], R_tmp[fold] = KCrossValidation( 
                                                                                fold, 
                                                                                train_features, 
                                                                                train_labels, 
                                                                                num_val_samples, 
                                                                                num_epochs, 
                                                                                batch_size, 
                                                                                verbose, 
                                                                                n1_start, 
                                                                                n2_start) 
                                                
    

    R_recent = sum(R_tmp)/len(R_tmp)
    mae_recent = sum(k_fold_mae)/len(k_fold_mae)


    dict_highest_R['R: {}, {}'.format(n1_start, n2_start)] = R_recent
    dict_lowest_MAE['MAE: {}, {}'.format(n1_start, n2_start)] = mae_recent


    best_history = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]
    
    print(mae_best, mae_recent, best_architecture)


    # 
    # Find the model with the lowest error
    i = 0
    for model in k_models :
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
