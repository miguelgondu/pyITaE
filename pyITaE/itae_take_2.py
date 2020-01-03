import numpy as np
import GPy

def load_map():
    return [], [], []

def deploy(controller):
    return 0, 0

def check_stopping_condition(performance, recorded_perfs):
    return False

def itae(path):
    perf_map, behaviors_map, controllers = load_map(path)

    real_map = perf_map.copy()
    next_index = np.argmax(real_map)

    next_controller = controllers[next_index]
    X = []
    Y = []
    recorded_perfs = []

    while True:
        performance, behavior = deploy(next_controller)
        recorded_perfs.append(performance)

        stopping_condition = check_stopping_condition(performance, recorded_perfs)

        dimension = len(behavior)
        kernel = GPy.kern.Matern52(dimension, lengthscale=1, ARD=False) + GPy.kern.White(dimension,np.sqrt(0.1))

        if stopping_condition:
            break

        X.append(behavior)
        Y.append(performance - perf_map[next_index])

        m = GPy.models.GPRegression(X, Y, kernel)
        mean, variance = m.predict(behaviors_map)

        real_map = mean + perf_map

        next_index = np.argmax(real_map)
        next_controller = controllers[next_index]
