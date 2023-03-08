
# ## Importing Data

# 
# -*- coding: utf-8 -*-
# Regression Example With Boston Dataset: Standardized and Wider
import os
import math
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from enum import unique
from pandas import read_csv
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import datasets

import dicts_functions
from  Data_to_CSV_Integrals_imports import transform_data
import aws_s3

import pandas as pd
from pandas import DataFrame

from multiprocessing import Process
from multiprocessing import Manager

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from psynlig import pca_loadings_map



def importData(data, scaler):

    train_dataset = data.sample(frac=0.8, random_state=5096)
    test_dataset = data.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Concentration')
    test_labels = test_features.pop('Concentration')

    train_features = dicts_functions.get_dict(scaler.fit_transform(train_features.to_numpy()))
    test_features = dicts_functions.get_dict(scaler.fit_transform(test_features.to_numpy()))

    #For later use
    data_labels = data.pop('Concentration')

    return data, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, 

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
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'], run_eagerly=True)
    
    return model #, early_stop

def PCA_Decomposition(data, std_scaler, components):

    Xstd = std_scaler.fit_transform(data)

    pca_data = PCA(n_components=components)
    Xpca  = pca_data.fit_transform(Xstd)

    print(f'Explained variation per principal component: {pca_data.explained_variance_ratio_}', sum(pca_data.explained_variance_ratio_))

    tmp = []
    for i in range(components):
        tmp.append(f'PCA {i}')

    principal_data_df = pd.DataFrame(data = Xpca , columns = tmp)

    return pca_data, Xpca, principal_data_df

def bubble(PCA, feature_names):
    plt.style.use('seaborn-talk')
    
    kwargs = {
        'heatmap': {
        'vmin': -1,
        'vmax': 1,
                    },
        }

    pca_loadings_map(
        PCA,
        feature_names,
        bubble=True,
        annotate=False,
        **kwargs
    )

    kwargs['heatmap']['vmin'] = 0
    pca_loadings_map(
        PCA,
        feature_names,
        bubble=True,
        annotate=False,
        plot_style='absolute',
        **kwargs
    )

    plt.show()

def euclidean(X, Xpca, feature_names):
    ccircle = []
    eucl_dist = []

    for i,j in enumerate(X .T):

        corr = [np.corrcoef(j, Xpca[:, z])[0,1] for z in range(11)]
        ccircle.append([sqr for sqr in corr])
        eucl_dist.append(np.sqrt(sum([sqr**2 for sqr in corr])))

    for _w, x_val in enumerate(X .T):
        for _y, z_val in enumerate(X .T):
            if (_w >= _y):
                continue

            with plt.style.context(('seaborn-whitegrid')):
                fig, axs = plt.subplots(figsize=(6, 6))
                for i,j in enumerate(eucl_dist):
                    arrow_col = plt.cm.cividis((eucl_dist[i] - np.array(eucl_dist).min())/\
                                            (np.array(eucl_dist).max() - np.array(eucl_dist).min()) )
                    axs.arrow(0,0, # Arrows start at the origin
                            ccircle[i][_w],  #0 for PC1
                            ccircle[i][_y],  #1 for PC2
                            lw = 2, # line width
                            length_includes_head=True, 
                            color = arrow_col,
                            fc = arrow_col,
                            head_width=0.05,
                            head_length=0.05)
                    axs.text(ccircle[i][_w]/2,ccircle[i][_y]/2, feature_names[i])
                # Draw the unit circle, for clarity
                circle = Circle((0, 0), 1, facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
                axs.add_patch(circle)
                axs.set_xlabel(f"PCA {_w + 1}")
                axs.set_ylabel(f"PCA {_y + 1}")
            plt.tight_layout()
            plt.show()

def KCrossValidation(i, features, labels, num_val_samples, epochs, batch, verbose, input_params, n1, n2, return_dict):

    val_data = features[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([features[:i * num_val_samples], features[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([labels[:i * num_val_samples], labels[(i + 1) * num_val_samples:]],     axis=0)

    model = build_model(input_params, n1, n2) #, early_stop = build_model(n1, n2)

    print('Training fold #', i)
    history = model.fit(
        partial_train_data, partial_train_targets,
        epochs=epochs, batch_size=batch, validation_split=0.3, verbose=verbose #, callbacks=early_stop
    )

    history = DataFrame(history.history)

    test_loss, test_mae, test_mse = model.evaluate(val_data, val_targets, verbose=verbose)
    test_R, y = Pearson(model, val_data, val_targets.to_numpy(), batch, verbose )

    model.save(f".\\WQM_NNs\\Model [{n1}, {n2}] {i}")

    return_dict[i] = (model.to_json(), model.get_weights(), history['val_mae'], test_mae, test_R)

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
        verbose=verbose_,
        workers=3,
        use_multiprocessing=True,
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
    #dataset = read_csv('aggregated_data.csv')
    
    ## DATA IMPORTING AND HANDLING

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
            df = df.append(transform_data(i),ignore_index=True,sort=False)

    dataset = shuffle(df)
    std_scaler = StandardScaler()

    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, = importData(dataset.copy(), std_scaler)
   
    # ## PRINCIPAL COMPONENT ANALYSIS
    num_components = 11 #Minimum: Time, current, derivative

    # pca_data, Xpca, principal_data_df = PCA_Decomposition(all_features.copy().to_numpy(), std_scaler, num_components)
    # bubble(pca_data, train_dataset.keys()[0:num_components])
    # euclidean(all_features.to_numpy(), Xpca, train_dataset.keys()[0:num_components])

    # ## NEURAL NETWORK PARAMETERS
    # 
    k_folds = 4
    num_val_samples = len(train_labels) // k_folds

    n1_start, n2_start = 8, 8
    sum_nodes = 17 #32

    num_epochs = 4 #400 #500
    batch_size = 500 #50
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

    #### Where the Magic Happens
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
                                                num_components,
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


            dict_highest_R[f'R: {j}, {i}'] = R_recent
            dict_lowest_MAE[f'MAE: {j}, {i}'] = mae_recent

            if (mae_recent <= mae_best):
                mae_best = mae_recent
                best_networks = k_models
                best_architecture = [j,i]
                best_history = [ np.mean([x[z] for x in k_mae_history]) for z in range(num_epochs)]
            
            print(mae_best, mae_recent, best_architecture)

    # Delete all other models here instead
    optimal_NNs  = best_networks
    i = 0

    print(best_architecture)
    
    filepath = r"C:\\Users\\junai\\AppData\\Roaming\\Python\\Python39\\Scripts\\Asparagine Machine Learning\\Additional Parameters\\WQM_NNs"
    local_download_path = os.path.expanduser(filepath)
    print(local_download_path)
    for filename in os.listdir(local_download_path):    
        if str(best_architecture) in filename:
            continue;

        if f"Model" in filename:
            print(filename)
            shutil.rmtree(filepath + "\\" + filename, ignore_errors=False)

        i +=1

    # Plotting Loss Transition
    smooth_mae_history = smooth_curve(best_history)

    #   _predictions =DataFrame({ key:pd.Series(value) for key, value in _predictions.items() })
    dict_epochs = { 
        "Epochs" : range(1, len(best_history) + 1),
        "Lowest MAE": best_history,
        "Smoothed Epochs": range(1, len(smooth_mae_history) + 1),
        "Lowest MAE Smoothed": smooth_mae_history,
        }
    

    zip_filename = "WQM_NNs"

    shutil.make_archive(zip_filename, 'zip', r".\\WQM_NNs")
    s3 = aws_s3.s3_bucket()
    
    dict_epochs = dict_epochs | dict_highest_R | dict_lowest_MAE
    dict_epochs = DataFrame({ key:pd.Series(value) for key, value in dict_epochs.items() })

    dict_epochs.to_csv(f'Evolution and Architecture PCA - Sum {sum_nodes} - Epochs {num_epochs} - Folds {k_folds}.csv')
        
