#load dataset 
#chia train, val, test 
#normalize 0,1 
#resize (if need) 
#def get_dataloader():
#    return train_loader, val_loader, test_loader

#Task 1.1 use multi features rather than just only Open 
     #load data AAPL.csv 
import pandas as pd 
import os 

def load_raw_data():
     base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
     file_path = os.path.join(base_path, "data", "AAPL.csv")

     df = pd.read_csv(file_path)
     return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(df.columns)
    
    #multi features 
from sklearn.preprocessing import MinMaxScaler 
import numpy as np 

def load_and_preprocessing(time_step=60):
     df = load_raw_data()
     features = ['Low', 'High', 'Open', 'Close', 'Adjusted Close', 'Volume']
     data = df[features].values
     scaler = MinMaxScaler()
     data_scaled = scaler.fit_transform(data)
     return data_scaled, scaler 