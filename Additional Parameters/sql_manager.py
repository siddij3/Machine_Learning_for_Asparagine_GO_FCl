import pandas as pd
from sqlite3 import connect
# import pymysql
# import mysql.connector as connection
from sqlalchemy import create_engine
from sqlalchemy import text

import sqlalchemy 

from  Data_to_CSV_Integrals_imports import transform_data
import file_management

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def get_creds():
    server = 'localhost' 
    database = 'wqm' 
    username = 'wqm_admin' 
    password = 'password'  
    port = 3306

    con = f'mysql+pymysql://{username}:{password}@{server}/{database}'
    return con

def connect():
    engine = create_engine(
            get_creds(), 
            pool_recycle=3600)

    return engine

def get_table_name():
    return 'asp_go'

def check_tables(engine, table):
    isTable = False

    query = text(f"SELECT * FROM {table}")

    with engine.begin() as conn:
        try:
            result = conn.execute(query)
        except:
            return isTable 
        
    isTable = True
    return result

def pandas_to_sql(table_name, pandas_dataset, engine):
    pandas_dataset.to_sql(table_name, con=engine)
    
def pandas_to_sql_if_exists(table_name, pandas_dataset, engine, action):
    pandas_dataset.to_sql(table_name, con=engine.connect(), if_exists=action)


def sql_to_pandas(table_name, engine):
    return pd.read_sql_table(table_name, con=engine.connect())


# data.to_sql(name=database, con=conn, if_exists = 'replace', index=False, flavor = 'mysql')
if __name__ == '__main__':

    engine = connect()
    table_name = 'asp_go'

    if (not check_tables(engine, table_name)):
        dataset = shuffle(file_management.create_df_from_csv())
    
        pandas_to_sql(table_name, dataset, engine)
        print(f"Created table {table_name} and contents")
    else:
        dataset = sql_to_pandas(table_name, engine)

    std_scaler = StandardScaler()
    all_features, data_labels, train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, std_scaler = file_management.importData(dataset.copy(), std_scaler)
    
    std_params = pd.DataFrame([std_scaler.mean_, std_scaler.scale_, std_scaler.var_], 
                       columns = train_features.keys())
    
    std_params['param_names'] = ['mean_', 'scale_', 'var_']

    table_name = 'std_params'
    if (not check_tables(engine, table_name)):
        pandas_to_sql(table_name, std_params, engine)
        print(f"Created table {table_name} and contents")
    else:
        print(pandas_to_sql_if_exists('std_params', std_params, engine, "replace"))



    print(std_params)



    # result_dataFrame = pd.read_sql(query, mydb)

