# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# ----------------- Récupération et normalisation des entrées ------------------- #

def calculate_date(str_date):
    month = int(str_date[0:2]) 
    n_days = 0
    for i in range(month-1):
        if i+1 in [1, 3, 5, 7, 8, 10, 12]:
            n_days += 31
        elif i+1 == 2:
            n_days += 28
        else:
            n_days += 30
    day = int(str_date[3:5])
    if month == 2 and day == 29:
        n_days += 28
    else:
        n_days += day
    return n_days-1


humidity = []
temp = []
rain_log = []
press = []
day_sin = []
day_cos = []
hour_sin = []
hour_cos = []
N = 0
with open("meteo_datas.csv", mode='r') as file:
    csvfile = csv.reader(file)
    next(csvfile)
    for line in csvfile:
        N += 1
        date = calculate_date(line[1][5:10]) 
        day_sin.append(np.sin(2*np.pi*date/365))
        day_cos.append(np.cos(2*np.pi*date/365))
        
        hour = int(line[2])
        hour_sin.append(np.sin(2*np.pi*hour/24))
        hour_cos.append(np.cos(2*np.pi*hour/24))    
        
        temperature = float(line[3])
        temp.append(temperature)
        
        pressure = float(line[4])
        press.append(pressure)
        
        hum = float(line[5])
        humidity.append(hum)
        
        if line[6] == "":
            r = 0.0
        else:
            r = float(line[6])
        log_r = np.log(1+r)
        rain_log.append(log_r)
   
humidity = np.array(humidity) / 100 
   
mu_T = np.mean(temp)
sig_T = np.std(temp)

mu_P = np.mean(press)
sig_P = np.std(press)

mu_r = np.mean(rain_log)
sig_r = np.std(rain_log)

temp = (np.array(temp) - mu_T) / sig_T
press = (np.array(press) - mu_P) / sig_P
rain_log = (np.array(rain_log) - mu_r) / sig_r

datas = np.array([np.array([hour_sin[i], hour_cos[i], day_sin[i], day_cos[i], humidity[i], temp[i], press[i], rain_log[i]]) for i in range(0, N)])
# -------------------------------- Création du LSTM ----------------------------- #

lookback = 48
n_inputs = 8
window_size = 24
X_train, Y_rain_train, Y_temp_train = [], [], []

for i in range(lookback, N-window_size):
    X_train.append(np.array(datas[i-lookback:i]))

for i in range(lookback, N-window_size):
    Y_rain_train.append(np.array(rain_log[i:i+window_size]))
    Y_temp_train.append(np.array(temp[i:i+window_size]))

X_train = np.array(X_train)
Y_rain_train = np.array(Y_rain_train)
Y_temp_train = np.array(Y_temp_train)

inputs = Input(shape=(lookback, n_inputs))

X = LSTM(64, return_sequences=False)(inputs)
X = Dropout(0.2)(X)
X = Dense(64, activation="relu")(X)

temp_output = Dense(window_size, name="temperature")(X)
rain_output = Dense(window_size, activation="sigmoid", name="rain")(X)

model = Model(inputs= inputs, outputs=[temp_output, rain_output])

model.compile(
        optimizer="adam",
        loss ={
                "temperature": "mse",
                "rain": "binary_crossentropy"
        },
        metrics={
            "temperature": "mae",
            "rain": "accuracy"
        }
    )   

model.summary()

model.fit(
    X_train,
    {"temperature": Y_temp_train,
     "rain": Y_rain_train},
    epochs = 20,
    batch_size = 32,
    validation_split = 0.2
    )

losses = model.evaluate(X_train, {"temperature": Y_temp_train, "rain": Y_rain_train})
print(losses)