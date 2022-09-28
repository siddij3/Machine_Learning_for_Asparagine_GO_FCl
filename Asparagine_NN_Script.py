# %% [markdown]
# ## Importing Data

# %%
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from sklearn.utils import shuffle
import math
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
from pandas import DataFrame

import tensorflow as tf
import numpy as np
import os

dataset = read_csv('aggregated_data.csv')
dataset = shuffle(dataset)

std_scaler = StandardScaler()

# %%
def importData(data, scaler):

    train_dataset = data.sample(frac=0.8, random_state=1234)
    test_dataset = data.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Concentration')
    test_labels = test_features.pop('Concentration')

    train_features = scaler.fit_transform(train_features.to_numpy())
    dict = {
        'Time':train_features[:, 0], 
        'Current':train_features[:, 1], 
        'Spin Coating':train_features[:, 2] ,
        'Increaing PPM':train_features[:, 3], 
        'Temperature':train_features[:, 4], 
        'Repeat Sensor Use':train_features[:, 5], 
        'Days Elapsed':train_features[:, 6]
        }
    train_features = DataFrame(dict)

    test_features = scaler.fit_transform(test_features.to_numpy())
    dict = {
        'Time':test_features[:, 0], 
        'Current':test_features[:, 1], 
        'Spin Coating':test_features[:, 2] ,
        'Increaing PPM':test_features[:, 3], 
        'Temperature':test_features[:, 4], 
        'Repeat Sensor Use':test_features[:, 5], 
        'Days Elapsed':test_features[:, 6]
        }
    test_features = DataFrame(dict)

    #For later use
    data_labels = data.pop('Concentration')

    return data, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, 


# %% [markdown]
# # Neural Network Creation and Selection Process

# %% [markdown]
# ### Functions: Build NN Model, Fit Model, K Cross Validation, Pearson Correlation Coefficient

# %%
def build_model(n1, n2):
  #Experiment with different models, thicknesses, layers, activation functions; Don't limit to only 10 nodes; Measure up to 64 nodes in 2 layers
  

    model = Sequential([
    layers.Dense(n1, activation=tf.nn.relu, input_shape=[7]),
    layers.Dense(n2, activation=tf.nn.relu),
    layers.Dense(1)
    ])

    optimizer = RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])
    #early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) Doesn't work with K-fold cross validation

    return model #, early_stop

def KCrossValidation(i, features, labels, num_val_samples, epochs, batch, verbose, n1, n2):

    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)

    model = build_model(n1, n2)

    print('Training fold #', i)
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, validation_split=0.3, verbose=verbose #, callbacks=early_stop
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    return model, history['val_mae'], test_loss, test_mae, test_mse, test_R

def Pearson(model, features, y_true, batch, verbose_):
    y_pred = model.predict(
        features,
        batch_size=batch,
        verbose=verbose_,
        workers=3,
        use_multiprocessing=False,
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

# %% [markdown]
# ## NEURAL NETWORK PARAMETERS

# %%
all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, = importData(dataset.copy(), std_scaler)
k_folds = 4
num_val_samples = len(train_labels) // k_folds

n1_start, n2_start = 5,5 #8
sum_nodes = 32 #32

num_epochs = 400 #500
batch_size = 32 #50
verbose = 0


print("\n")
print("Number Folds: ", k_folds)
print("Initial Layers N1: ", n1_start)
print("Initial Layers N2: ", n2_start)
print("Total Layers: ", sum_nodes)
print("Epochs: ", num_epochs)
print("Batch Size: ", batch_size)
print("\n")

best_architecture = [0,0]

dict_lowest_MAE,dict_highest_R  = {}, {}
best_mae_networks, best_mae_history = 0,0,

mae_best  = 10

# %% [markdown]
# #### Where the Magic Happens

# %%
#(STEPS FROM DEEP LEARNING WITH PYTHON BY MANNING)

for i in range(n1_start, sum_nodes):

    for j in range(n2_start, sum_nodes):
        if (i+j > sum_nodes):
            continue
        
        print("first hidden layer", j)
        print("second hidden layer", i)
        k_fold_mae, k_models, k_mae_history, R_tmp = [None]*k_folds, [None]*k_folds, [None]*k_folds, [None]*k_folds

        for fold in range(k_folds):
            model, val_mae_history, test_loss, test_mae, test_mse, test_R = KCrossValidation(
                fold, 
                train_features, 
                train_labels, 
                num_val_samples, 
                num_epochs, 
                batch_size, 
                verbose, 
                j, 
                i)
            
            R_tmp[fold] = test_R
            k_fold_mae[fold] = test_mae
            
            k_models[fold] = model

            k_mae_history[fold] = val_mae_history


        R_recent = sum(R_tmp)/len(R_tmp)
        dict_highest_R['R: {}, {}'.format(j, i)] = R_recent


        mae_recent = sum(k_fold_mae)/len(k_fold_mae)

        dict_lowest_MAE['MAE: {}, {}'.format(j, i)] = mae_recent

        if (mae_recent <= mae_best):
            mae_best = mae_recent
            best_mae_networks = k_models
            best_architecture = [j,i]
            best_mae_history = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]
        
        print(mae_best, mae_recent, best_architecture)


print("Models done")
# %%
# Find the model with the lowest error
optimal_NNs_mae = best_mae_networks

# %% [markdown]
# Plotting Loss Transition

def smooth_curve(points, factor=0.7):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

print(best_architecture)

smooth_mae_history = smooth_curve(best_mae_history)

#   _predictions =DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
dict_epochs = { 
    "Epochs" : range(1, len(best_mae_history) + 1),
    "Lowest MAE": best_mae_history,

    "Smoothed Epochs": range(1, len(smooth_mae_history) + 1),

    "Lowest MAE Smoothed": smooth_mae_history,
    "Smoothed Epochs": range(1, len(smooth_mae_history) + 1)

    }

dict_lowest_MAE = DataFrame(dict_lowest_MAE, index = [0])
dict_highest_R = DataFrame(dict_highest_R, index = [0])
df_R_MAE = dict_highest_R.append(dict_lowest_MAE, ignore_index=True)

dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })
dict_epochs  = dict_epochs.append(df_R_MAE, ignore_index=True)
dict_epochs.to_csv('epochs - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)

# %% [markdown]
# Scaling Data Set Function

def scaleDataset(data):
    data = std_scaler.fit_transform(data.to_numpy())
    dict = {
        'Time':data[:, 0], 
        'Current':data[:, 1], 
        'Spin Coating':data[:, 2] ,
        'Increaing PPM':data[:, 3], 
        'Temperature':data[:, 4], 
        'Repeat Sensor Use':data[:, 5], 
        'Days Elapsed':data[:, 6]
        }
    return DataFrame(dict)


# %% [markdown]
# ## Functions for Isolating Parameters

def isolateParam(optimal_NNs, data, parameter, start_index, end_index, NN_start, batch, verbose, mae_or_R): #Somethign wrong in here TODO
    # Split the data labels with time
    param_index= [np.where(data[parameter].to_numpy()  == i)[0] for i in range(start_index, end_index)]

    scaled_features = scaleDataset(all_features.copy())
    #The full features of the data points that use certain time values
    param_features =  [scaled_features.iloc[param_index[i]] for i in range(start_index, end_index)]
    param_labels = [data_labels.to_numpy()[param_index[i]] for i in range(start_index, end_index)]

    mae, R = [], []
    _predictions = {}

    for i in range(NN_start, end_index):
        tmp_mae, tmp_R = [], []
        
        j = 0
        for NN in optimal_NNs:
            test_loss, test_mae, test_mse = NN.evaluate(
                param_features[i], 
                param_labels[i], 
                batch_size=batch,  
                verbose=verbose
                )

            tmp, tmp_predictions = Pearson(NN, param_features[i], param_labels[i], batch, verbose) 
            tmp_R.append(tmp)

            dict_title_real = "Real NN {} Correlation for {} - {}: {}".format(j, parameter, i, mae_or_R)
            dict_title = "Predicted NN {} Correlation for {} - {}: {}".format(j, parameter, i, mae_or_R)

            _predictions[dict_title_real] = param_labels[i].tolist()
            _predictions[dict_title] = tmp_predictions.tolist()

    
            tmp_mae.append(test_mae)
            j += 1

        R.append(tmp_R)
        mae.append(tmp_mae)

    

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv('Optimal {} Isolated {} - Sum {} - Epochs {} - Folds {}.csv'.format(mae_or_R, parameter, sum_nodes, num_epochs, k_folds), index=False)
    
    average_R = [sum(i)/len(i) for i in R]
    average_mae = [sum(i)/len(i) for i in mae]

    return average_R, average_mae

# %%
def IsolateBinaryTime(optimal_NNs, data, parameter, start_time, batch, vbs, mae_or_R):
    # Splitting Spin Coating, then seperating by time
    data = data.dropna()

    ss_1 = np.where(data[parameter].to_numpy()  ==  1)[0]
    ss_0 = np.where(data[parameter].to_numpy()  ==  0)[0]

    times_index, shared_time_1, shared_time_0 = [], [], []


    for i in range(0, 51):
        times_index.append(np.where(data['Time'].to_numpy()  == i)[0].tolist())

        time_1_tmp = [index_sc for index_sc in ss_1 if index_sc in times_index[i]]
        time_0_tmp = [index_sc for index_sc in ss_0 if index_sc in times_index[i]]

                        
        shared_time_1.append(time_1_tmp)
        shared_time_0.append(time_0_tmp)
  

    scaled_features = scaleDataset(all_features.copy())

    shared_features0 = [scaled_features.iloc[shared_time_0[i]] for i in range(0,51)] 
    shared_features1 = [scaled_features.iloc[shared_time_1[i]] for i in range(0,51)]
    shared_features = [shared_features0, shared_features1]

    shared_labels0 = [data_labels.to_numpy()[shared_time_0[i]] for i in range(0,51)] 
    shared_labels1 = [data_labels.to_numpy()[shared_time_1[i]] for i in range(0,51)]
    shared_labels = [shared_labels0, shared_labels1]


    shared_mae, shared_R = [], []
    _predictions = {}

    i = start_time
    for i in range(start_time, 51):
        sc_tmp_mae, sc_tmp_R = [], []


        for j in range(0, 2):
            tmp_mae, tmp_R = [], []

            k = 0
            for NN in optimal_NNs:

                test_loss, test_mae, test_mse = NN.evaluate(
                    shared_features[j][i], 
                    shared_labels[j][i],
                    batch_size=batch,  
                    verbose=vbs
                    )

                tmp, tmp_predictions = Pearson(NN, shared_features[j][i], shared_labels[j][i], batch, verbose) 
                tmp_R.append(tmp)

                tmp_mae.append(test_mae)

                dict_title_real = "Real NN {} Correlation for T {}, {} {}: {}".format(k, i, parameter, j, mae_or_R)
                dict_title = "Predicted NN {} Correlation for T {}, {} {}: {}".format(k, i, parameter, j, mae_or_R)

                _predictions[dict_title_real] = shared_labels[j][i].tolist()
                _predictions[dict_title] = tmp_predictions.tolist()
                k+=1

            sc_tmp_mae.append(tmp_mae)
            sc_tmp_R.append(tmp_R)


        shared_mae.append(sc_tmp_mae)
        shared_R.append(sc_tmp_R)



    _predictions =DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv('Optimal {} Isolated Time and {} - Sum {} - Epochs {} - Folds {}.csv'.format(mae_or_R, parameter, sum_nodes, num_epochs, k_folds), index=False)
    

    averages_R, averages_mae = [], []
    for i in shared_mae:
        averages_mae.append([sum(i[0])/len(i[0]), sum(i[1])/len(i[1])])

    for i in shared_R:
        averages_R.append([sum(i[0])/len(i[0]), sum(i[1])/len(i[1])])

    return averages_R, averages_mae


def repeatSensor(optimal_NNs, data, parameter1, parameter2, start_index, end_index, start_time, batch, vbs, mae_or_R):

    # Split the data labels with RSU
    repeat_index= [np.where(data[parameter1].to_numpy()  == i+1)[0] for i in range(start_index, end_index)]

    shared_tr_1, shared_tr_2, shared_tr_3 = [], [], []

    times_index = []
    for i in range(0, 51):
        times_index.append(np.where(data[parameter2].to_numpy()  == i)[0].tolist())

        tr_1_tmp, tr_2_tmp, tr_3_tmp = [], [], []

        for j in range(len(repeat_index)):
    
            for index_123 in repeat_index[j]:

                if index_123 in times_index[i] and j == 0:
                    tr_1_tmp.append(index_123)
                elif index_123 in times_index[i] and j == 1:
                    tr_2_tmp.append(index_123)
                elif index_123 in times_index[i] and j == 2:
                    tr_3_tmp.append(index_123)

        shared_tr_1.append(tr_1_tmp)
        shared_tr_2.append(tr_2_tmp)
        shared_tr_3.append(tr_3_tmp)

    scaled_features = scaleDataset(all_features.copy())
    #The full features of the data points that use certain time values
    tr_features = []
    tr_labels = []


    for i in range(0, 51):
        tr_features.append([
            scaled_features.iloc[shared_tr_1[i]], 
            scaled_features.iloc[shared_tr_2[i]], 
            scaled_features.iloc[shared_tr_3[i]]
            ])

        tr_labels.append([
            data_labels.to_numpy()[shared_tr_1[i]], 
            data_labels.to_numpy()[shared_tr_2[i]], 
            data_labels.to_numpy()[shared_tr_3[i]]
            ])



    tr_mae = []
    tr_R = []
    _predictions = {}
    for i in range(start_time, 51):
        tr_tmp_mae, tr_tmp_R = [], []


        for j in range(start_index, end_index):
            tmp_mae, tmp_R = [], []
            k = 0

            for NN in optimal_NNs:
                test_loss, test_mae, test_mse = NN.evaluate(tr_features[i][j], tr_labels[i][j], batch_size=batch,  verbose=vbs)
                

                tmp, tmp_predictions = Pearson(NN, tr_features[i][j], tr_labels[i][j], batch, verbose) 
                tmp_R.append(tmp)

                tmp_mae.append(test_mae)


                dict_title_real = "Real NN {} Correlation for T {}, Repeat {}: {} ".format(k, i, j,  mae_or_R)
                dict_title = "Predicted NN {} Correlation for T {}, Repeat {}: {} ".format(k, i, j,  mae_or_R)
                
                _predictions[dict_title_real] = tr_labels[i][j].tolist()
                _predictions[dict_title] = tmp_predictions.tolist()
                k+=1

            tr_tmp_mae.append(tmp_mae)
            tr_tmp_R.append(tmp_R)

        tr_mae.append(tr_tmp_mae)
        tr_R.append(tr_tmp_R)


    averages_mae = []
    averages_R = []

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv('Optimal {} - Isolated {} and {} - Sum {} - Epochs {} - Folds {}.csv'.format(mae_or_R, parameter1, parameter2, sum_nodes, num_epochs, k_folds), index=False)

    for i in tr_mae:
        averages_mae.append([sum(i[0])/len(i[0]), sum(i[1])/len(i[1]), sum(i[2])/len(i[2])])

    for i in tr_R:
        averages_R.append([sum(i[0])/len(i[0]), sum(i[1])/len(i[1]), sum(i[2])/len(i[2])])

    return averages_R, averages_mae

# %%
start_index= 0
end_index = 3
vbs = 1
start_time = 1
param_batches = 10

str_MAE = "MAE"

# %% [markdown]
# #### Repeat Sensor Use
print("Isolating Repeat Sensor Use and Time")

R_of_tr_mae, mae_of_tr_mae = repeatSensor(
    optimal_NNs_mae, 
    dataset, 
    'Repeat Sensor Use', 
    'Time',
   start_index, 
    end_index, 
    start_time, 
    param_batches, 
    vbs, 
    str_MAE
    )


# %% [markdown]
# #### Isolating Increasing PPM and Time
print("Isolating Increasing PPM and Time")
R_of_increasing_mae, mae_of_increasing_mae = IsolateBinaryTime(optimal_NNs_mae, dataset, 'Increasing PPM', start_time, param_batches, vbs, str_MAE)
#R_of_increasing_testdata, mae_of_increasing_testdata = IsolateBinaryTime(optimal_NNs_mae, test_dataset, 'Increasing PPM', start_time, param_batches, vbs, "TEST DATA")


# %% [markdown]
# #### Isolating Spin Coating
print("Isolating Spin Coating")
R_of_sc_mae, mae_of_sc_mae = isolateParam(optimal_NNs_mae, dataset, 'Spin Coating', 0, 2, 0, param_batches, vbs, str_MAE)
#R_of_sc_testdata, mae_of_sc_testdata = isolateParam(optimal_NNs_mae, test_dataset, 'Spin Coating', 0, 2, 0, param_batches, vbs, "TEST DATA")

# %% [markdown]
# #### Isolating Time
print("Isolating Time")

R_time_mae, mae_averages_time_mae = isolateParam(optimal_NNs_mae, dataset, 'Time', 0, 51, start_time, param_batches, vbs, str_MAE)
#R_time_testdata, mae_averages_time_testdata = isolateParam(optimal_NNs_mae, test_dataset, 'Time', 0, 51, start_time, param_batches, vbs, "TEST DATA")

# %% [markdown]
# #### Isolating Spin Coating and Time
print("Isolating Spin Coating and Time")
R_of_sct_mae, mae_of_sct_mae = IsolateBinaryTime(optimal_NNs_mae, dataset, 'Spin Coating', start_time, param_batches, vbs, str_MAE)
#R_of_sct_testdata, mae_of_sct_testdata = IsolateBinaryTime(optimal_NNs_mae, test_dataset, 'Spin Coating', start_time, param_batches, vbs, "TEST DATA")




# %% [markdown]
# # Printing to CSV


dict_all = {
    "SC: R"    : R_of_sc_mae,
    "SC: MAE"  : mae_of_sc_mae,
    #"SC Test: R"    : R_of_sc_testdata,
    #"SC Test: MAE"  : mae_of_sc_testdata,

    "Time: Test R"    : R_time_mae, 
    "Time: Test MAE"  : mae_averages_time_mae, 
    #"Time: Test R"    : R_time_testdata, 
    #"Time: Test MAE"  : mae_averages_time_testdata, 

    "Time SC: 0;: R"    : [i[0] for i in R_of_sct_mae], 
    "Time SC: 1: R"    : [i[1] for i in R_of_sct_mae],     
    "Time SC: 0 : MAE"  : [i[0] for i in mae_of_sct_mae], 
    "Time SC: 1 : MAE"  : [i[1] for i in mae_of_sct_mae], 

    #"Time SC: 0; Test R"    : [i[0] for i in R_of_sct_testdata], 
    #"Time SC: 1; Test R"    : [i[1] for i in R_of_sct_testdata],     
    #"Time SC: 0; Test MAE"  : [i[0] for i in mae_of_sct_testdata], 
    #"Time SC: 1; Test MAE"  : [i[1] for i in mae_of_sct_testdata], 
    
    "Time Increasing: 0 : R"    : [i[0] for i in R_of_increasing_mae], 
    "Time Increasing: 1 : R"    : [i[1] for i in R_of_increasing_mae],     
    "Time Increasing: 0 : MAE"  : [i[0] for i in mae_of_increasing_mae] , 
    "Time Increasing: 1 : MAE"  : [i[1] for i in mae_of_increasing_mae], 

    #"Time Increasing: 0; Test R"    : [i[0] for i in R_of_increasing_testdata], 
    #"Time Increasing: 1; Test R"    : [i[1] for i in R_of_increasing_testdata],     
    #"Time Increasing: 0; Test MAE"  : [i[0] for i in mae_of_increasing_testdata] , 
    #"Time Increasing: 1; Test MAE"  : [i[1] for i in mae_of_increasing_testdata], 

    "Day 1 : R"    : [i[0] for i in R_of_tr_mae], 
    "Day 2 : R"    : [i[1] for i in R_of_tr_mae], 
    "Day 3 : R"    : [i[2] for i in R_of_tr_mae], 

    "Day 1 : MAE"    : [i[0] for i in mae_of_tr_mae], 
    "Day 2 : MAE"    : [i[1] for i in mae_of_tr_mae], 
    "Day 3 : MAE"    : [i[2] for i in mae_of_tr_mae],     
    
 #   "Day 1 : Test R"    : [i[0] for i in R_of_tr_testdata], 
 #   "Day 2 : Test R"    : [i[1] for i in R_of_tr_testdata], 
  #  "Day 3 : Test R"    : [i[2] for i in R_of_tr_testdata], 

  #  "Day 1 : Test MAE"    : [i[0] for i in mae_of_tr_testdata], 
 #   "Day 2 : Test MAE"    : [i[1] for i in mae_of_tr_testdata], 
  #  "Day 3 : Test MAE"    : [i[2] for i in mae_of_tr_testdata]

    }

dict_all = DataFrame({ key:pd.Series(value) for key, value in dict_all.items() })
dict_all.to_csv('Final MAE and R  - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)

print(best_architecture)
