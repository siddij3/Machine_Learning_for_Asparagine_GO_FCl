
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
import math
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import Sequential

from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import dicts_functions
from  Data_to_CSV_Integrals_imports import transform_data
import aws_s3

import sql_manager
from sql_manager import pandas_to_sql
from sql_manager import sql_to_pandas
from sql_manager import pandas_to_sql_if_exists

import file_management

import pandas as pd
from pandas import DataFrame

from multiprocessing import Process
from multiprocessing import Manager

import tensorflow as tf
import numpy as np

# # Neural Network Creation and Selection Process
# 
def build_model(input, n1, n2):
  #Experiment with different models, thicknesses, layers, activation functions; Don't limit to only 10 nodes; Measure up to 64 nodes in 2 layers
  
    model = Sequential([
    layers.Dense(n1, activation=tf.nn.relu, input_shape=[input]),
    layers.Dense(n2, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse']) #, run_eagerly=True)
    
    return model #, early_stop

def KCrossValidation(i, features, labels, num_val_samples, epochs, batch, verbose, input_params, n1, n2, return_dict, folder_name):

    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)

    model = build_model(input_params, n1, n2) #, early_stop = build_model(n1, n2)

    print('Training fold #', i)
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, 
        validation_split=0.3, verbose=verbose,
        workers=3,
        use_multiprocessing=True,
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    model.save(f".\\{folder_name}\\Model [{n1}, {n2}] {i}")

    return_dict[i] = (history['val_mae'], test_mae, test_R)

def Pearson(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_,
        workers=3,
        use_multiprocessing=True
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

    table_name = sql_manager.get_table_name()
    engine = sql_manager.connect()

    # if the table doesn't exist, create it from the csv file, 
    # and send the file to 
    if (not sql_manager.check_tables(engine, table_name)):
        dataset = shuffle(file_management.create_df_from_csv())

        pandas_to_sql(table_name, dataset, engine)
    else:
        dataset = sql_to_pandas(table_name, engine)

    std_scaler = StandardScaler()
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, std_scaler = file_management.importData(dataset.copy(), std_scaler)
    
    std_params = pd.DataFrame([std_scaler.mean_, std_scaler.scale_, std_scaler.var_], 
                       columns = train_features.keys())
    std_params['param_names'] = ['mean_', 'scale_', 'var_']

    table_name = 'std_params'
    if (not sql_manager.check_tables(engine, table_name)):
        pandas_to_sql(table_name, std_params, engine)
    else:
        pandas_to_sql_if_exists('std_params', std_params, engine, "replace")



    # ## PRINCIPAL COMPONENT ANALYSIS
    num_components = 11 #Minimum: Time, current, derivative

    # ## NEURAL NETWORK PARAMETERS
    # 
    k_folds = 2
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 8, 8
    sum_nodes = 16 #32

    num_epochs = 10 #400 #500
    batch_size = 16 #50
    verbose = 0

    folder_name = file_management.get_file_path()


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

    #### Where the Magic Happens
    for i in range(n2_start, sum_nodes):
        for j in range(n1_start, sum_nodes):
            if (i+j > sum_nodes):
                continue
            
            print("first hidden layer", j)
            print("second hidden layer", i)
            k_fold_mae, k_mae_history, R_tmp =  [None]*k_folds, [None]*k_folds, [None]*k_folds

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
                                                num_components,
                                                j, 
                                                i, return_dict,
                                                folder_name))
                _futures[fold].start()   
                
            for job in _futures:
                job.join()

        # ( history['val_mae'], test_mae, test_mse, test_R)
            for fold in range(k_folds):
                k_mae_history[fold] = return_dict.values()[fold][0]
                k_fold_mae[fold] = return_dict.values()[fold][1]
                R_tmp[fold] = return_dict.values()[fold][2]


            R_recent = sum(R_tmp)/len(R_tmp)
            mae_recent = sum(k_fold_mae)/len(k_fold_mae)


            dict_highest_R[f'R: {j}, {i}'] = R_recent
            dict_lowest_MAE[f'MAE: {j}, {i}'] = mae_recent

            if (mae_recent <= mae_best):
                mae_best = mae_recent
                best_architecture = [j,i]
                best_history = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]
            
            print(mae_best, mae_recent, best_architecture)

    # Delete all other models here instead
    optimal_NNs  = best_networks
    i = 0

    print(best_architecture)
    
    filepath = file_management.get_file_path()
    local_download_path = os.path.expanduser(filepath)
    print(local_download_path)

    for filename in os.listdir(local_download_path):    
        if str(best_architecture) in filename:
            continue;

        if f"Model" in filename:
            print(filename)
            shutil.rmtree(f".\\{filepath}\\{filename}", ignore_errors=False)

        i +=1
