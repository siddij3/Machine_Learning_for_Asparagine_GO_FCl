import dicts_functions

from Data_to_CSV_Integrals_imports import transform_data
import os

def get_file_path():
    folder_name = "WQM_NNs_test"
    return f".\\{folder_name}"

def get_data_path():
    folder_name = "Data"
    return f".\\{folder_name}"

def create_df_from_csv():
    filepath = get_data_path()
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

    return df

def importData(data, scaler):

    train_dataset = data.sample(frac=0.8, random_state=5096)
    test_dataset = data.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Concentration')
    test_labels = test_features.pop('Concentration')

    try:
        train_features = train_features.drop(['index'], axis = 1)
        test_features = test_features.drop(['index'], axis = 1)
    except:
        print("'index' parameter does not exist");

    train_features = dicts_functions.get_dict(scaler.fit_transform(train_features.to_numpy()))
    test_features = dicts_functions.get_dict(scaler.transform(test_features.to_numpy()))

    #For later use
    data_labels = data.pop('Concentration')

    return data, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels,  scaler
