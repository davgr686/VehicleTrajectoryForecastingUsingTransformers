from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os
import pandas as pd

def preprocess_dataset(data, max = 10, max_len = 40, input_len = 30, use_existing_scalar, scaler_filepath):
    global scaler
    if (use_existing_scalar):
      scaler = joblib.load(scaler_filepath) 
      data[['X_REL', 'Y_REL']] = scaler.transform(data[['X_REL', 'Y_REL']])
    else:
      scaler = MinMaxScaler(feature_range=(0, max))
      data[['X_REL', 'Y_REL']] = scaler.fit_transform(data[['X_REL', 'Y_REL']])

    unique_peds = data['Vehicle_ID'].unique()
    unique_peds = sorted(unique_peds)
    inputs = []
    outputs = []
    indexes = []
    for ped in unique_peds:
        if (len(data[data['Vehicle_ID'] == ped]) >= max_len):
          seq_inner = []
          indexes_inner = []
          i = 0
          for indx, row in data[data['Vehicle_ID'] == ped].iterrows():
            x = round(row['X_REL'])
            y = round(row['Y_REL'])
            ## Cantor pairing function:
            bin = y * max + x
            i += 1
            if i == max_len:
              break
            seq_inner.append(int(bin))
            indexes_inner.append(ped)
          inputs.append([seq_inner[0:input_len]])
          outputs.append([seq_inner[input_len + 1:]])
          indexes.append([indexes_inner])

    train_inputs, test_inputs, train_targets, test_targets, train_indx, test_indx = train_test_split(inputs, outputs, indexes, train_size=0.1, random_state=0)
    return torch.tensor(train_inputs), torch.tensor(test_inputs), torch.tensor(train_targets), torch.tensor(test_targets), train_indx, test_indx
