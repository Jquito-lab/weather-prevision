# -*- coding: utf-8 -*-

import neural_network as neur
import csv
import numpy as np



def get_input_output(i: int):
    inp = np.concatenate([[day_sin[j], day_cos[j], hour_sin[j], hour_cos[j], temp[j], press[j], humidity[j], rain_log[j]]for j in range(i-23, i+1)])
    outp = np.concatenate([[temp[j], rain_log[j]] for j in range(i+1, i+25)])
    return inp, outp

inputs = [np.concatenate([[day_sin[j], day_cos[j], hour_sin[j], hour_cos[j], temp[j], press[j], humidity[j], rain_log[j]]for j in range(l-23, l+1)]) for l in range(23, N-25)]

outputs = [np.concatenate([[temp[j], rain_log[j]] for j in range(l+1, l+25)]) for l in range(23, N-25)]
nn_meteo = neur.create_nn([192, 64, 32, 48])

neur.heetal_init(nn_meteo)

neur.grad_descent(nn_meteo, inputs, outputs, 32)
