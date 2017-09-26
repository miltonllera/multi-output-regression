import numpy as np


def load_boston_housing_data():
    data = []
    with open('data.data', mode='rb') as f:
        for line in f.readlines():
            data.append(line)

    return np.asarray(data)
