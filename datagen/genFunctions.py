import pandas as pd
import numpy as np

# GenFunction Class
class GenFunction():

    def __init__(self):
        return

    def regular_sin(self, numPoints, frequency, stopTime, startTime=0,
                    noise=0):
        time_vec = np.linspace(startTime, stopTime, numPoints)
        if noise == 0:
            data = np.sin(2*np.pi*frequency*time_vec)
        else:
            noise_arr = np.random.normal(loc=0, scale=noise, size=numPoints)
            data = np.sin(2*np.pi*frequency*time_vec) + noise_arr
        data_rolled = np.roll(data, -1)[:-1]
        data_mat = np.matrix(data[:-1]).T
        resp_mat = np.matrix(data_rolled).T
        return data_mat, resp_mat, time_vec

    def irregular_sin(self, numPoints, numSelPoints, frequency, stopTime,
                      startTime=0, noise=0):
        time = np.linspace(startTime, stopTime, numPoints)
        if noise == 0:
            data = np.sin(2*np.pi*frequency*time)
        else:
            noise_arr = np.random.normal(loc=0, scale=noise, size=numPoints)
            data = np.sin(2*np.pi*frequency*time_vec) + noise_arr
        index = np.sort(np.random.choice(range(numPoints), size=numSelPoints,
                        replace=False))
        time_irr = time[index]
        data_irr = data[index]
        delta_t = [0] + list(np.array(time_irr[1:]) - np.array(time_irr[:-1]))
        data_mat = np.matrix([list(data_irr), delta_t]).T
        data_mat = data_mat[:-1, :]
        resp_mat = np.matrix(np.roll(data_irr, -1)[:-1]).T
        time_vec = time_irr[1:]
        return data_mat, resp_mat, time_vec

    def irregular_sin_features(self, numPoints, numSelPoints, frequency,
                               stopTime, startTime=0, noise=0):
        time = np.linspace(0, stopTime, numPoints)
        if noise == 0:
            data = np.sin(2*np.pi*frequency*time)
        else:
            noise_arr = np.random.normal(loc=0, scale=noise, size=numPoints)
            data = np.sin(2*np.pi*frequency*time_vec) + noise_arr
        index = np.sort(np.random.choice(range(numPoints), size=numSelPoints,
                        replace=False))
        time_irr = time[index]
        data_irr = data[index]
        delta_t = [0] + list(np.array(time_irr[1:]) - np.array(time_irr[:-1]))
        derivative = np.divide(np.diff(data_irr), np.diff(time_irr))
        magDerivative = np.lib.pad(derivative, (1, 0), 'constant',
                                   constant_values=(0, 0))
        dDerivative = np.lib.pad(np.diff(magDerivative), (1, 0), 'constant',
                                 constant_values=(0, 0))
        data_mat = np.matrix([list(data_irr), delta_t, list(magDerivative),
                             list(dDerivative)]).T
        data_mat = data_mat[:-1, :]
        resp_mat = np.matrix(np.roll(data_irr, -1)[:-1]).T
        time_vec = time_irr[1:]
        return data_mat, resp_mat, time_vec

    def regular_mv(self, numPoints, L, noise=0):
        mean = np.zeros(numPoints)
        vec = np.array([[j for j in range(0, numPoints)] for k in
                        range(0, start=numPoints)])
        covMat = np.exp(-(vec - vec.T)**2/float(L**2))
        if noise == 0:
            data = np.random.multivariate_normal(mean, covMat, size=(1,))[0]
        else:
            noise_arr = np.random.normal(loc=0, scale=noise, size=numPoints)
            data = np.random.multivariate_normal(mean, covMat, size=(1,))[0] +\
                    noise_arr
        data_rolled = np.roll(data, -1)[:-1]
        data_mat = np.matrix(data[:-1]).T
        resp_mat = np.matrix(data_rolled).T
        return data_mat, resp_mat, time_vec

    def irregular_mv(self, numPoints, numSelPoints, L, noise=0):
        mean = np.zeros(numPoints)
        vec = np.array([[j for j in range(0, numPoints)] for k in
                        range(0, start=numPoints)])
        covMat = np.exp(-(vec - vec.T)**2/float(L**2))
        if noise == 0:
            data = np.random.multivariate_normal(mean, covMat, size=(1,))[0]
        else:
            noise_arr = np.random.normal(loc=0, scale=noise, size=numPoints)
            data = np.random.multivariate_normal(mean, covMat, size=(1,))[0] +\
                    noise_arr
        time = np.array(range(numPoints))
        index = np.sort(np.random.choice(range(numPoints), size=numSelPoints,
                        replace=False))
        time_irr = time[index]
        data_irr = data[index]
        delta_t = [0] + list(np.array(time_irr[1:]) - np.array(time_irr[:-1]))
        data_mat = np.matrix([list(data_irr), delta_t]).T
        data_mat = data_mat[:-1, :]
        resp_mat = np.matrix(np.roll(data_irr, -1)[:-1]).T
        time_vec = time_irr[1:]
        return data_mat, resp_mat, time_vec

    def irregular_mv_features(self, numPoints, numSelPoints, L, noise=0):
        mean = np.zeros(numPoints)
        vec = np.array([[j for j in range(0, numPoints)] for k in
                        range(0, start=numPoints)])
        covMat = np.exp(-(vec - vec.T)**2/float(L**2))
        if noise == 0:
            data = np.random.multivariate_normal(mean, covMat, size=(1,))[0]
        else:
            noise_arr = np.random.normal(loc=0, scale=noise, size=numPoints)
            data = np.random.multivariate_normal(mean, covMat, size=(1,))[0] +\
                    noise_arr
        time = np.array(range(numPoints))
        index = np.sort(np.random.choice(range(numPoints), size=numSelPoints,
                        replace=False))
        time_irr = time[index]
        data_irr = data[index]
        delta_t = [0] + list(np.array(time_irr[1:]) - np.array(time_irr[:-1]))
        derivative = np.divide(np.diff(data_irr), np.diff(time_irr))
        magDerivative = np.lib.pad(derivative, (1, 0), 'constant',
                                   constant_values=(0, 0))
        dDerivative = np.lib.pad(np.diff(magDerivative), (1, 0), 'constant',
                                 constant_values=(0, 0))
        data_mat = np.matrix([list(data_irr), delta_t, list(magDerivative),
                             list(dDerivative)]).T
        data_mat = data_mat[:-1, :]
        resp_mat = np.matrix(np.roll(data_irr, -1)[:-1]).T
        time_vec = time_irr[1:]
        return data_mat, resp_mat, time_vec
