# -*- coding: utf-8 -*-

import neural_network as neur
import numpy as np

nn_or = neur.create_nn([2, 4, 1])

inputs = [[np.random.randint(0,2), np.random.randint(0,2)] for i in range(100000)]
outputs = [int(inputs[i][0] or inputs[i][1]) for i in range(10000)]