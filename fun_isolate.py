from pandas import DataFrame

def scaleDataset(data, std_scaler):
    
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
        tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds

        avg_predictions = [None]*k_folds
        
        j = 0
        for NN in optimal_NNs:
            test_loss, test_mae, test_mse = NN.evaluate(
                param_features[i], 
                param_labels[i], 
                batch_size=batch,  
                verbose=verbose
                )

            tmp, tmp_predictions = Pearson(NN, param_features[i], param_labels[i], batch, verbose) 
            tmp_R[j] = tmp

            dict_title_real = "Real NN {} Correlation for {} - {}: {}".format(j, parameter, i, mae_or_R)
            dict_title = "Predicted NN {} Correlation for {} - {}: {}".format(j, parameter, i, mae_or_R)

            _predictions[dict_title_real] = param_labels[i].tolist()
            _predictions[dict_title] = tmp_predictions.tolist()

            avg_predictions[j] = tmp_predictions.tolist()
    
            tmp_mae[j] = test_mae
            j += 1

        dict_average = "Averages for {}:".format(i)
        arr_avg_predictions = np.transpose(avg_predictions)
        _predictions[dict_average] = [np.mean(i) for i in arr_avg_predictions]


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
            tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds
            k = 0

            avg_predictions = [None]*k_folds

            for NN in optimal_NNs:
                test_loss, test_mae, test_mse = NN.evaluate(tr_features[i][j], tr_labels[i][j], batch_size=batch,  verbose=vbs)
                

                tmp, tmp_predictions = Pearson(NN, tr_features[i][j], tr_labels[i][j], batch, verbose) 

                tmp_R[k] = tmp
                tmp_mae[k] = test_mae


                dict_title_real = "Real NN {} Correlation for T {}, Repeat {}: {} ".format(k, i, j,  mae_or_R)
                dict_title = "Predicted NN {} Correlation for T {}, Repeat {}: {} ".format(k, i, j,  mae_or_R)
                
                _predictions[dict_title_real] = tr_labels[i][j].tolist()
                _predictions[dict_title] = tmp_predictions.tolist()

                avg_predictions[k] = tmp_predictions.tolist()
                k+=1

            dict_average = "Averages for T {} - Repeat {}:".format(i, j)
            arr_avg_predictions = np.transpose(avg_predictions)
            _predictions[dict_average] = [np.mean(z) for z in arr_avg_predictions]

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


def daysElapsed(optimal_NNs, data, parameter,  batch, verbose, mae_or_R): 
    # Split the data labels with time

    unique_val = np.unique(data[parameter].to_numpy())
    param_index = [np.where(data[parameter].to_numpy()  == i)[0] for i in unique_val]
    
    shared_tr_1, shared_tr_2, shared_tr_3, shared_tr_4 = [], [], [], []
    days = [None] * len(param_index)
    times_index = []

    for i in range(0, 51):
        times_index.append(np.where(data["Time"].to_numpy()  == i)[0].tolist())

        tr_1_tmp, tr_2_tmp, tr_3_tmp, tr_4_tmp  = [], [], [], []

        for j in range(len(param_index)):
    
            for index_123 in param_index[j]:

                if index_123 in times_index[i] and j == unique_val[0]:
                    tr_1_tmp.append(index_123)
                elif index_123 in times_index[i] and j == unique_val[1]:
                    tr_2_tmp.append(index_123)
                elif index_123 in times_index[i] and j == unique_val[2]:
                    tr_3_tmp.append(index_123)
                elif index_123 in times_index[i] and j == unique_val[3]:
                    tr_3_tmp.append(index_123)

        shared_tr_1.append(tr_1_tmp)
        shared_tr_2.append(tr_2_tmp)
        shared_tr_3.append(tr_3_tmp)
        shared_tr_4.append(tr_4_tmp)

    scaled_features = scaleDataset(all_features.copy())
    #The full features of the data points that use certain time values
    tr_features = []
    tr_labels = []


    for i in range(0, 51):
        tr_features.append([
            scaled_features.iloc[shared_tr_1[i]], 
            scaled_features.iloc[shared_tr_2[i]], 
            scaled_features.iloc[shared_tr_3[i]],
            scaled_features.iloc[shared_tr_4[i]],
            ])

        tr_labels.append([
            data_labels.to_numpy()[shared_tr_1[i]], 
            data_labels.to_numpy()[shared_tr_2[i]], 
            data_labels.to_numpy()[shared_tr_3[i]],
            data_labels.to_numpy()[shared_tr_4[i]]
            ])



    tr_mae = []
    tr_R = []
    _predictions = {}
    for i in range(start_time, 51):
        tr_tmp_mae, tr_tmp_R = [], []


        for j, val in enumerate(unique_val):
            tmp_mae, tmp_R = [None]*k_folds, [None]*k_folds
            k = 0

            print(j, val)

            avg_predictions = [None]*k_folds

            for NN in optimal_NNs:
                test_loss, test_mae, test_mse = NN.evaluate(tr_features[i][j], tr_labels[i][j], batch_size=batch,  verbose=vbs)
                

                tmp, tmp_predictions = Pearson(NN, tr_features[i][j], tr_labels[i][j], batch, verbose) 

                tmp_R[k] = tmp
                tmp_mae[k] = test_mae

                

                dict_title_real = "Real NN {} Correlation for T {}, Day Elapsed {}: {} ".format(k, i, val,  mae_or_R)
                dict_title = "Predicted NN {} Correlation for T {}, Days Elapsed {}: {} ".format(k, i, val,  mae_or_R)
                
                _predictions[dict_title_real] = tr_labels[i][j].tolist()
                _predictions[dict_title] = tmp_predictions.tolist()

                avg_predictions[k] = tmp_predictions.tolist()
                k+=1

            dict_average = "Averages for T {} - Repeat {}:".format(i, j)
            arr_avg_predictions = np.transpose(avg_predictions)
            _predictions[dict_average] = [np.mean(z) for z in arr_avg_predictions]

            tr_tmp_mae.append(tmp_mae)
            tr_tmp_R.append(tmp_R)

        tr_mae.append(tr_tmp_mae)
        tr_R.append(tr_tmp_R)


    averages_mae = []
    averages_R = []

    _predictions = DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    _predictions.to_csv('Optimal {} - Isolated {} and Time - Sum {} - Epochs {} - Folds {}.csv'.format(mae_or_R, parameter, sum_nodes, num_epochs, k_folds), index=False)

    for i in tr_mae:
        averages_mae.append([sum(i[0])/len(i[0]), sum(i[1])/len(i[1]), sum(i[2])/len(i[2]), sum(i[3])/len(i[3])])

    for i in tr_R:
        averages_R.append([sum(i[0])/len(i[0]), sum(i[1])/len(i[1]), sum(i[2])/len(i[2]), sum(i[3])/len(i[3])])

    return averages_R, averages_mae

# %%
start_index= 0
end_index = 3
vbs = 1
start_time = 1
param_batches = 10

str_MAE = "MAE"

print("Days Elapsed")                   
R_of_days, mae_of_days = daysElapsed(optimal_NNs_mae, dataset, 'Days Elapsed',  param_batches, vbs, str_MAE)


# %% [markdown]
# #### Isolating Increasing PPM and Time
print("Isolating Increasing PPM and Time")
#R_of_increasing_mae, mae_of_increasing_mae = IsolateBinaryTime(optimal_NNs_mae, dataset, 'Increasing PPM', start_time, param_batches, vbs, str_MAE)
#R_of_increasing_testdata, mae_of_increasing_testdata = IsolateBinaryTime(optimal_NNs_mae, test_dataset, 'Increasing PPM', start_time, param_batches, vbs, "TEST DATA")


# %% [markdown]
# #### Isolating Spin Coating
print("Isolating Spin Coating")
#R_of_sc_mae, mae_of_sc_mae = isolateParam(optimal_NNs_mae, dataset, 'Spin Coating', 0, 2, 0, param_batches, vbs, str_MAE)
#R_of_sc_testdata, mae_of_sc_testdata = isolateParam(optimal_NNs_mae, test_dataset, 'Spin Coating', 0, 2, 0, param_batches, vbs, "TEST DATA")

# %% [markdown]
# #### Isolating Time
print("Isolating Time")

#R_time_mae, mae_averages_time_mae = isolateParam(optimal_NNs_mae, dataset, 'Time', 0, 51, start_time, param_batches, vbs, str_MAE)
#R_time_testdata, mae_averages_time_testdata = isolateParam(optimal_NNs_mae, test_dataset, 'Time', 0, 51, start_time, param_batches, vbs, "TEST DATA")

# %% [markdown]
# #### Isolating Spin Coating and Time
print("Isolating Spin Coating and Time")
#R_of_sct_mae, mae_of_sct_mae = IsolateBinaryTime(optimal_NNs_mae, dataset, 'Spin Coating', start_time, param_batches, vbs, str_MAE)
#R_of_sct_testdata, mae_of_sct_testdata = IsolateBinaryTime(optimal_NNs_mae, test_dataset, 'Spin Coating', start_time, param_batches, vbs, "TEST DATA")

# %% [markdown]
# #### Repeat Sensor Use
print("Isolating Repeat Sensor Use and Time")

#R_of_tr_mae, mae_of_tr_mae = repeatSensor(
 #   optimal_NNs_mae, 
 #   dataset, 
 #   'Repeat Sensor Use', 
 #   'Time',
 #  start_index, 
 #   end_index, 
 #   start_time, 
 #   param_batches, 
 #   vbs, 
 #   str_MAE
 #   )


# %% [markdown]
# # Printing to CSV


dict_all = {
    "SC: R"    : R_of_sc_mae,
    "SC: MAE"  : mae_of_sc_mae,
    "SC Test: R"    : R_of_sc_testdata,
    "SC Test: MAE"  : mae_of_sc_testdata,

    "Time": [i for i in range(1, 51)],
    "Time: Test R"    : R_time_mae, 
    "Time: Test MAE"  : mae_averages_time_mae, 
    "Time: Test R"    : R_time_testdata, 
    "Time: Test MAE"  : mae_averages_time_testdata, 

    "Time SC: 0;: R"    : [i[0] for i in R_of_sct_mae], 
    "Time SC: 1: R"    : [i[1] for i in R_of_sct_mae],     
    "Time SC: 0 : MAE"  : [i[0] for i in mae_of_sct_mae], 
    "Time SC: 1 : MAE"  : [i[1] for i in mae_of_sct_mae], 

    "Time SC: 0; Test R"    : [i[0] for i in R_of_sct_testdata], 
    "Time SC: 1; Test R"    : [i[1] for i in R_of_sct_testdata],     
    "Time SC: 0; Test MAE"  : [i[0] for i in mae_of_sct_testdata], 
    "Time SC: 1; Test MAE"  : [i[1] for i in mae_of_sct_testdata], 

    "Time Increasing: 0 : R"    : [i[0] for i in R_of_increasing_mae], 
    "Time Increasing: 1 : R"    : [i[1] for i in R_of_increasing_mae],     
    "Time Increasing: 0 : MAE"  : [i[0] for i in mae_of_increasing_mae] , 
    "Time Increasing: 1 : MAE"  : [i[1] for i in mae_of_increasing_mae], 

    "Time Increasing: 0; Test R"    : [i[0] for i in R_of_increasing_testdata], 
    "Time Increasing: 1; Test R"    : [i[1] for i in R_of_increasing_testdata],     
    "Time Increasing: 0; Test MAE"  : [i[0] for i in mae_of_increasing_testdata] , 
    "Time Increasing: 1; Test MAE"  : [i[1] for i in mae_of_increasing_testdata], 

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
