# %% [markdown]
# ## Importing Data

# %%
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
from pandas import read_csv
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import math
import pandas as pd
from pandas import DataFrame


import numpy as np


dataset = read_csv('.\Data\\aggregated_data.csv')
dataset = shuffle(dataset)

std_scaler = StandardScaler()

# %%
def importData(data, scaler):

    train_dataset = data.sample(frac=0.8, random_state=5096)
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

n1_start, n2_start = 6,6
sum_nodes = 32 #32

num_epochs = 400 #500
batch_size = 32 #50
verbose = 0

#filepath = '.\\epochs {} - Sum {} - Epochs {} - Batch {} - Data {}\\'.format(num_epochs, sum_nodes, batch_size, "both")

path = ".\\"
optimal_NNs = [
    load_model(f'{path}Model [23, 7] number 0'), 
    load_model(f'{path}Model [23, 7] number 1'), 
    load_model(f'{path}Model [23, 7] number 2'), 
    load_model(f'{path}Model [23, 7] number 3') ]

print("\n")
print("path: ", path)
print("Model: Model [23, 7]")
print("\n")
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
            tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds
            avg_predictions = [None]*k_folds

            k = 0
            for NN in optimal_NNs:

                test_loss, test_mae, test_mse = NN.evaluate(
                    shared_features[j][i], 
                    shared_labels[j][i],
                    batch_size=batch,  
                    verbose=vbs
                    )

                tmp, tmp_predictions = Pearson(NN, shared_features[j][i], shared_labels[j][i], batch, verbose) 
                tmp_R[k] = tmp
                tmp_mae[k] = test_mae

                dict_title_real = "Real NN {} Correlation for T {}, {} {}: {}".format(k, i, parameter, j, mae_or_R)
                dict_title = "Predicted NN {} Correlation for T {}, {} {}: {}".format(k, i, parameter, j, mae_or_R)

                _predictions[dict_title_real] = shared_labels[j][i].tolist()
                _predictions[dict_title] = tmp_predictions.tolist()

                avg_predictions[k] = tmp_predictions.tolist()
                k+=1

            dict_average = "Averages for T {} - {} {}:".format(i, parameter, j)
            arr_avg_predictions = np.transpose(avg_predictions)
            _predictions[dict_average] = [np.mean(z) for z in arr_avg_predictions]

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
            tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds
            avg_predictions = [None]*k_folds

            k = 0
            for NN in optimal_NNs:

                test_loss, test_mae, test_mse = NN.evaluate(
                    shared_features[j][i], 
                    shared_labels[j][i],
                    batch_size=batch,  
                    verbose=vbs
                    )

                tmp, tmp_predictions = Pearson(NN, shared_features[j][i], shared_labels[j][i], batch, verbose) 
                tmp_R[k] = tmp
                tmp_mae[k] = test_mae

                dict_title_real = "Real NN {} Correlation for T {}, {} {}: {}".format(k, i, parameter, j, mae_or_R)
                dict_title = "Predicted NN {} Correlation for T {}, {} {}: {}".format(k, i, parameter, j, mae_or_R)

                _predictions[dict_title_real] = shared_labels[j][i].tolist()
                _predictions[dict_title] = tmp_predictions.tolist()

                avg_predictions[k] = tmp_predictions.tolist()
                k+=1

            dict_average = "Averages for T {} - {} {}:".format(i, parameter, j)
            arr_avg_predictions = np.transpose(avg_predictions)
            _predictions[dict_average] = [np.mean(z) for z in arr_avg_predictions]

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

def repeatSensor(optimal_NNs, data, parameter1, parameter2, batch, vbs, isTest):
    

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
            tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds

            avg_predictions = [None]*k_folds

            for k, NN in enumerate(optimal_NNs):
                test_loss, test_mae, test_mse = NN.evaluate(rsu_vals, labels[i][j], batch_size=batch,  verbose=vbs)
                

                tmp, tmp_predictions = Pearson(NN, rsu_vals, labels[i][j], batch, verbose) 

                tmp_R[k] = tmp
                tmp_mae[k] = test_mae


                dict_title_real = "Real NN {} Correlation for T {}, Repeat {}: {} ".format(k, i, unique_vals_inner[j],   isTest)
                dict_title = "Predicted NN {} Correlation for T {}, Repeat {}: {} ".format(k, i, unique_vals_inner[j],   isTest)
                
                _predictions[dict_title_real] = labels[i][j].tolist()
                _predictions[dict_title] = tmp_predictions.tolist()

                avg_predictions[k] = tmp_predictions.tolist()


            dict_average = "Averages for T {} - Repeat {}:".format(i, j)
            arr_avg_predictions = np.transpose(avg_predictions)
            _predictions[dict_average] = [np.mean(z) for z in arr_avg_predictions]

            tr_tmp_mae.append(sum(tmp_mae)/len(tmp_mae))
            tr_tmp_R.append(sum(tmp_R)/len(tmp_R))

        tr_mae.append(tr_tmp_mae)
        tr_R.append(tr_tmp_R)



    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv('Optimal - Isolated {} and {} - Sum {} - Epochs {} - Folds {}.csv'.format( parameter1, parameter2, sum_nodes, num_epochs, k_folds), index=False)

 
    averages_mae = [[j for j in i] for i in tr_mae] 
    averages_R = [[j for j in i] for i in tr_R] 

    return averages_R, averages_mae


def daysElapsed(optimal_NNs, data, param1, param2,  batch, verbose): 
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

            for sc, isSpin in enumerate(days):

                tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds
                avg_predictions = [None]*k_folds
                
                for k, NN in enumerate(optimal_NNs):
                    test_loss, test_mae, test_mse = NN.evaluate(
                                                isSpin, 
                                                labels[t][d][sc],
                                                batch_size=batch,  
                                                verbose=vbs
                                                )

                    
                    tmp, tmp_predictions = Pearson(NN, isSpin, labels[t][d][sc], batch, verbose) 
                    tmp_R[k] = tmp
                    tmp_mae[k] = test_mae

                    dict_title_real = "Real NN {} Correlation for T {}, Day {}: SC {}".format(k, t,  unique_vals_days[d], sc)
                    dict_title = "Predicted NN {} Correlation for T {}, Day {}: SC {}".format(k, t, unique_vals_days[d], sc)

                    _predictions[dict_title_real] = labels[t][d][sc].tolist()
                    _predictions[dict_title] = tmp_predictions.tolist()

                    avg_predictions[k] = tmp_predictions.tolist()


                dict_average = "Averages for T {} - Days {} SC {}:".format(t, d, sc)
                arr_avg_predictions = np.transpose(avg_predictions)
                _predictions[dict_average] = [np.mean(z) for z in arr_avg_predictions]

                sc_tmp_mae.append(sum(tmp_mae)/len(tmp_mae))
                sc_tmp_R.append(sum(tmp_R)/len(tmp_R))

            days_tmp_mae.append(sc_tmp_mae)
            days_tmp_R.append(sc_tmp_R)

        shared_mae.append(days_tmp_mae)
        shared_R.append(days_tmp_R)

    
    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv(f'Optimal {param1} - Isolated {param2} - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv', index=False)

    R_ = {}
    MAE_  = {}

    for d, days in enumerate(shared_R[0]):
        for sc, isSC in enumerate(shared_R[0][d]):
            tmp_time_R, tmp_time_MAE = [], []

            tmp_time_R = [ time[d][sc] for t, time in enumerate(shared_R)]
            tmp_time_MAE = [ shared_mae[t][d][sc]  for t, time in enumerate(shared_R)]

            MAE_title = "ED {}: SC {}; MAE".format(unique_vals_days[d], sc)
            R_title = "ED {}: SC {}; R".format(unique_vals_days[d], sc)
            
            MAE_[MAE_title] = tmp_time_MAE
            R_[R_title] = tmp_time_R

    R_ = DataFrame({  key:pd.Series(value) for key, value in R_.items() })
    MAE_ = DataFrame({ key:pd.Series(value) for key, value in MAE_.items() })
    R_MAE = pd.concat([R_, MAE_])

    return R_MAE#non_sc_days, sc_days

# %%


start_index= 0
end_index = 3
vbs = 1
start_time = 1
param_batches = 1

str_MAE = "MAE"


print("Free Chlorine and Time")
R_of_increasing , mae_of_increasing  = repeatSensor(optimal_NNs, dataset, 'Increasing PPM', 'Time', param_batches, vbs, str )


print("Isolating Repeat Sensor Use and Time")
R_of_tr , mae_of_tr  = repeatSensor(optimal_NNs, dataset, 'Repeat Sensor Use', 'Time', param_batches, vbs, str)

print("Days Elapsed")                   
dict_daysElapsed = daysElapsed(optimal_NNs, dataset, 'Spin Coating', 'Days Elapsed',  param_batches, vbs)

dict_all = {


    "SC: R"    : 4 ,
    #"SC: MAE"  : mae_of_sc ,
    #"SC Test: R"    : R_of_sc_testdata,
    #"SC Test: MAE"  : mae_of_sc_testdata,

    #"Time": [i for i in range(1, 51)],
    #"Time:  R"    : R_time , 
    #"Time:  MAE"  : mae_averages_time , 
    #"Time: Test R"    : R_time_testdata, 
    #"Time: Test MAE"  : mae_averages_time_testdata, 

    #"Time SC: 0;: R"    : [i[0] for i in R_of_sct ], 
    #"Time SC: 1: R"    : [i[1] for i in R_of_sct ],     
    #"Time SC: 0 : MAE"  : [i[0] for i in mae_of_sct ], 
    #"Time SC: 1 : MAE"  : [i[1] for i in mae_of_sct ], 

    #"Time SC: 0; Test R"    : [i[0] for i in R_of_sct_testdata], 
    #"Time SC: 1; Test R"    : [i[1] for i in R_of_sct_testdata],     
    #"Time SC: 0; Test MAE"  : [i[0] for i in mae_of_sct_testdata], 
    #"Time SC: 1; Test MAE"  : [i[1] for i in mae_of_sct_testdata], 

    "Time Increasing: 0 : R"    : [i[0] for i in R_of_increasing ], 
    "Time Increasing: 1 : R"    : [i[1] for i in R_of_increasing ],     
    "Time Increasing: 0 : MAE"  : [i[0] for i in mae_of_increasing ] , 
    "Time Increasing: 1 : MAE"  : [i[1] for i in mae_of_increasing ], 

    #"Time Increasing: 0; Test R"    : [i[0] for i in R_of_increasing_testdata], 
    #"Time Increasing: 1; Test R"    : [i[1] for i in R_of_increasing_testdata],     
    #"Time Increasing: 0; Test MAE"  : [i[0] for i in mae_of_increasing_testdata] , 
    #"Time Increasing: 1; Test MAE"  : [i[1] for i in mae_of_increasing_testdata], 

    "Day 1 : R"    : [i[0] for i in R_of_tr ], 
    "Day 2 : R"    : [i[1] for i in R_of_tr ], 
    "Day 3 : R"    : [i[2] for i in R_of_tr ], 

    "Day 1 : MAE"    : [i[0] for i in mae_of_tr ], 
    "Day 2 : MAE"    : [i[1] for i in mae_of_tr ], 
    "Day 3 : MAE"    : [i[2] for i in mae_of_tr ],     
 #   "Day 1 : Test R"    : [i[0] for i in R_of_tr_testdata], 
 #   "Day 2 : Test R"    : [i[1] for i in R_of_tr_testdata], 
  #  "Day 3 : Test R"    : [i[2] for i in R_of_tr_testdata], 

  #  "Day 1 : Test MAE"    : [i[0] for i in mae_of_tr_testdata], 
 #   "Day 2 : Test MAE"    : [i[1] for i in mae_of_tr_testdata], 
  #  "Day 3 : Test MAE"    : [i[2] for i in mae_of_tr_testdata]

    }

dict_all = DataFrame({ key:pd.Series(value) for key, value in dict_all.items() })
dict_all = pd.concat([dict_all, dict_daysElapsed])
dict_all.to_csv('Final MAE and R  - Sum {} - Epochs {} - Folds {}.csv'.format(sum_nodes, num_epochs, k_folds), index=False)
