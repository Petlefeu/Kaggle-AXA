#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import signal
import logging
import random
import os
import itertools as it
from pdb import set_trace as st
import sklearn as sk
from sklearn import metrics, cross_validation, svm, linear_model, ensemble, cluster
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

nt = 200                         # Number of trips per driver
csv_dir = "drivers"
pickles_dir = "pickles"
min_false_trips = 60                          # False trips (from other drivers) we add to the correct ones (from the driver of iterest)
max_false_trips = 80
n_noise = 10                          # Number of false trips we label as 'correct' for the training
n_correct_trips = 200                         # Number of correct trips in each dataset

n_inits_train = 20                          # Number of random datasets buit for train
n_inits_test = 5                           # Number of random datasets buit for test

batch_size = 800                         # Number of drivers to process in one batch

nd_test = 50                          # Number of drivers to use for AUC computation
drivers = os.listdir('drivers')       # List of driver names ('1', '10', '100', ...)
nd = len(drivers)
trips = map(str, range(1, nt+1))     # List of trip names (for one driver) ('1', '2', '3', ..., '200')

nb_process = 8

# ######################################
# #####            Utils           #####
# ######################################

percentile = lambda x, k: np.percentile(x, k) if x is not None and x.size else 0
mean = lambda x: np.mean(x) if x is not None and x.size else 0


def moving_average(a, n=3):
    if len(a.shape) == 1:
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        mvavg = ret[n - 1:] / n
    else:
        ret = np.cumsum(a, dtype=float, axis=0)
        ret[n:, :] = ret[n:, :] - ret[:-n, :]
        mvavg = ret[n - 1:, :] / n
    return mvavg

# ######################################
# ## Pre-loading data into pickles  ####
# ######################################


def preload_into_pickles():
    trip_pointers = []
    for i, d in enumerate(drivers):
        logging.info("Loading driver %d of %d into pickle." % (i, nd))
        driver_trips = [np.genfromtxt("%s/%s/%s.csv" % (csv_dir, d, t), skip_header=True, delimiter=',') for t in trips]
        trip_lengths = [len(t) for t in driver_trips]
        trip_pointers.append(np.cumsum(trip_lengths))
        np.save("%s/driver-%s.npy" % (pickles_dir, d), np.concatenate(driver_trips))
    np.save("%s/pointers.npy" % pickles_dir, np.array(trip_pointers))


def read_trip_pointers():
    trip_pointers = np.load("%s/pointers.npy" % pickles_dir)
    return trip_pointers


def read_from_pickles(drivers_batch):
    driver_arrays = {}
    N = len(drivers_batch)
    for i, d in enumerate(drivers_batch):
        logging.debug("Reading driver %d (#%s) from pickle (%d of %d)." % (d, drivers[d], i, N))
        array = np.load("%s/driver-%s.npy" % (pickles_dir, drivers[d]))
        driver_arrays[d] = array
    return driver_arrays

# ######################################
# #####     Features computation   #####
# ######################################


def compute_features(trip, d_id=None, t_id=None):
    # print d_id,t_id
    trip_duration = float(len(trip))
    speed, acceleration = speeds_and_accelerations(trip)
    speed = abs(speed)
    trip_length = float(np.sum(speed))
    speedup = acceleration[acceleration > 0]
    slowdown = -acceleration[acceleration < 0]
    try:
        curvature = get_curvature(trip)
    except:
        logging.exception('Cannot compute curvature for trip %d of driver %d' % (t_id, d_id))
        curvature = np.array([])

    curvature_percentiles = [percentile(curvature, pc) for pc in [90, 75, 50, 25, 10]]
    curvature_maxima_x = sp.signal.argrelmax(curvature, order=2)[0]
    curvature_maxima_y = curvature[curvature_maxima_x]
    curvature_maxima_speed = speed[curvature_maxima_x]
    curvature_maxima_percentiles = [percentile(curvature_maxima_y, pc) for pc in [75, 50, 25]]
    centrifugal_acceleration = np.power(curvature_maxima_speed, 2)*curvature_maxima_y
    # curvature_inter_maxima = [np.diff(maxima) for maxima in curvature_maxima]
    # median_curvature_inter_maxima = [np.percentile(cim, 50) if cim.size else trip_length for cim in curvature_inter_maxima]

    mean_slowdown, mean_speedup = speeds_near_stops(speed)

    features = [
        trip_length,
        trip_duration,
        np.percentile(speed, 75),
        np.percentile(speed, 50),
        np.percentile(speed, 25),
        np.percentile(speedup, 75),
        np.percentile(speedup, 50),
        np.percentile(speedup, 25),
        np.percentile(slowdown, 75),
        np.percentile(slowdown, 50),
        np.percentile(slowdown, 25),
        curvature_percentiles[0],
        curvature_percentiles[1],
        curvature_percentiles[2],
        curvature_percentiles[3],
        curvature_percentiles[4],
        len(curvature_maxima_x)/trip_length,
        curvature_maxima_percentiles[0],
        curvature_maxima_percentiles[1],
        curvature_maxima_percentiles[2],
        mean(centrifugal_acceleration),
        mean(curvature_maxima_speed),
        gps_signal(speed),
        mean_slowdown,
        mean_speedup,
    ]
    if not np.all(np.isfinite(features)):
        st()
    return features


def gps_signal(speed):
    return len(speed[speed > 56])


def speeds_and_accelerations(trip):
    speed_vectors = np.diff(trip, axis=0)
    speeds = np.linalg.norm(speed_vectors, axis=1)
    accelerations = np.diff(speeds, axis=0)
    return speeds, accelerations


# Return the curvature curve of the road
def get_curvature(trip):
    x_orig, y_orig = trip[:, 0], trip[:, 1]
    mask = ((np.diff(x_orig) != 0) | (np.diff(y_orig) != 0))
    x_orig, y_orig = x_orig[mask], y_orig[mask]
    try:
        trip_interpol, u = sp.interpolate.splprep([x_orig, y_orig])
    except:
        try:
            trip_interpol, u = sp.interpolate.splprep([x_orig, y_orig], k=2)
        except:
            try:
                trip_interpol, u = sp.interpolate.splprep([x_orig, y_orig], k=1)
            except:
                return np.array([0])

    u_new = np.linspace(0, 1, 100)
    x, y = sp.interpolate.splev(u_new, trip_interpol)
    dx, dy = sp.interpolate.splev(u_new, trip_interpol, der=1)
    d2x, d2y = sp.interpolate.splev(u_new, trip_interpol, der=2)
    # curvature = np.abs((dx*d2y-dy*d2x)/np.power(x, 3))
    curvature = np.abs(dx*d2y-dy*d2x)/np.power(np.power(dx, 2)+np.power(dy, 2), 1.5)
    return curvature[np.isfinite(curvature)]


# Ajout constance
# Return speedup and slowdown mean near stops
def speeds_near_stops(speeds):
    time_stop = 3               # time (in secund) minimal in the same position to have a stop state
    nb_slowdown = 0
    nb_speedup = 0
    mean_slowdown = 0
    mean_speedup = 0
    nb_stops = 0

    same_speed = 0

    for i in range(len(speeds)):
        if speeds[i] < 2:
            same_speed += 1
            if same_speed == time_stop:
                t_1 = i - time_stop
                t_2 = t_1 - 1
                t_3 = t_2 - 1
                if (t_1 > 1) and (speeds[t_3] > speeds[t_2] > speeds[t_1]):
                    nb_stops += 1
                    nb_slowdown += 1
                    mean_slowdown += speeds[t_3]*0.5 + speeds[t_2]*0.3 + speeds[t_1]*0.2
        else:
            if (same_speed >= time_stop):
                t_1 = i
                t_2 = t_1 + 1
                t_3 = t_2 + 1
                if (t_3 < len(speeds)) and (speeds[t_1] < speeds[t_2] < speeds[t_3]):
                    nb_speedup += 1
                    mean_speedup += speeds[t_3]*0.5 + speeds[t_2]*0.3 + speeds[t_1]*0.2
            same_speed = 0
    mean_slowdown = float(mean_slowdown)
    mean_speedup = float(mean_speedup)
    nb_stops = float(nb_stops)
    if (nb_slowdown != 0):
        mean_slowdown = mean_slowdown / nb_slowdown
    if (nb_speedup != 0):
        mean_speedup = mean_speedup / nb_speedup
    # print "Nombre de stop: %s" % nb_stops

    # print "Moyenne ralentissement %s, et acceleration %s" % (mean_slowdown,mean_speedup)
    return mean_slowdown, mean_speedup

# ######################################
# #####     Building data set      #####
# ######################################


# Returns the trip 't' of a driver 'd' (already converted into features)
# 'd' ranges from 0 to nd-1
# 't' ranges from 0 to nt-1
def get_trip(d, t, driver_arrays):
    try:
        return compute_features(get_raw_trip(d, t, driver_arrays), d_id=d, t_id=t)
    except:
        logging.exception('Cannot compute features for trip %d of driver %d' % (d, t))


def get_raw_trip(d, t, driver_arrays):
    t = int(t)
    begin = 0 if t == 0 else trip_pointers[d, t-1]
    end = None if t == nt else trip_pointers[d, t]
    raw_trip = driver_arrays[d][begin:end, :]
    return raw_trip


# Builds a dataset for a specific driver (with some correct trips and some false ones, at random)
def build_driver_train_dataset(driver, correct_trips, drivers_batch, driver_arrays):
    # Generating false trips
    n_false_trips = random.randint(min_false_trips, max_false_trips)
    logging.debug("Building dataset with %d correct trips and %d false ones" % (len(correct_trips), n_false_trips))
    false_drivers = random.sample(drivers_batch, n_false_trips)
    false_trips = [get_trip(d, random.randint(0, nt-1), driver_arrays) for d in false_drivers]

    # Building data
    X = np.array(correct_trips+false_trips)                                            # One row = 1 trip (each column is a feature)
    Y_true = np.concatenate([np.ones(len(correct_trips)), np.zeros(n_false_trips)])    # Target variable (1 if correct driver, 0 otherwise)
    Y_noisy = Y_true.copy()
    Y_noisy[n_correct_trips+np.random.randint(0, n_false_trips-1, n_noise)] = 1         # Adding noise (false trips labelled as true)
    X, Y_noisy, Y_true = sk.utils.shuffle(X, Y_noisy, Y_true)

    return X, Y_noisy, Y_true

# ######################################
# #####     Training & testing     #####
# ######################################


# Trains a classifier for the given driver
def train(driver, correct_trips, drivers_batch, driver_arrays):
    # Subsampling correct_trips
    correct_trips = random.sample(correct_trips, n_correct_trips)

    # Getting dataset
    X, Y_noisy, _ = build_driver_train_dataset(driver, correct_trips, drivers_batch, driver_arrays)

    # Training
    # cls = sk.linear_model.LogisticRegression(C=1000.0)
    # cls = sk.svm.SVC(C=0.001, probability=True)
    cls = sk.ensemble.RandomForestClassifier(n_estimators=200, max_features=None)
    #cls = sk.ensemble.AdaBoostClassifier()
    cls.fit(X, Y_noisy)

    return cls


def train_test_driver(d, driver_ids_batch, driver_arrays):
    # Building correct trips
    correct_trips = [get_trip(d, t, driver_arrays) for t in range(nt)]

    # Getting classifiers
    classifiers = [train(d, correct_trips, driver_ids_batch, driver_arrays) for i in range(n_inits_train)]

    # Predicting on new random datasets
    datasets = [build_driver_train_dataset(d, correct_trips, driver_ids_batch, driver_arrays) for i in range(n_inits_test)]
    Xs = [d_[0] for d_ in datasets]
    Ys_true = [d_[2] for d_ in datasets]
    Y_preds = [np.array([cls.predict_proba(X)[:, 1] for cls in classifiers]) for X in Xs]
    Y_preds_aggreg = [ys.mean(0) for ys in Y_preds]

    return Ys_true, Y_preds_aggreg


def train_predict_driver(d, drivers_batch, driver_arrays):
    # Building correct trips
    correct_trips = [get_trip(d, t, driver_arrays) for t in range(nt)]

    # Getting classifiers
    classifiers = [train(d, correct_trips, drivers_batch, driver_arrays) for i in range(n_inits_train)]

    # Predicting on all the driver's trips
    X = np.array(correct_trips)
    Y_preds = np.array([cls.predict_proba(X)[:, 1] for cls in classifiers])
    Y_preds_aggreg = Y_preds.mean(0)

    return Y_preds_aggreg


#########################################################
#####              Pre-loading data                 #####
#########################################################
# preload_into_pickles()


#########################################################
##    Computing final predictions for all drivers     ###
#########################################################

# MULTI PROCESS

def process_loop(driver_ids_batch, ids, nb_process, n_process):
    j = n_process
    Y_preds = []
    driver_arrays = read_from_pickles(driver_ids_batch)

    for d in driver_ids_batch:
        logging.info("Processing driver %d (%d of %d)" % (d, j, nd))
        try:
            Y_pred = train_predict_driver(d, driver_ids_batch, driver_arrays)
        except:
            logging.exception('Failed !')
            Y_pred = -np.ones(nt)
        Y_preds += Y_pred.tolist()
        j += nb_process

    result = np.array([ids, Y_preds]).T
    np.savetxt("submission_%s.csv" % n_process, result, comments="", fmt="%s", delimiter=",")

trip_pointers = read_trip_pointers()
driver_ids_to_process, driver_names_to_process = sk.utils.shuffle(range(nd), drivers)
ids = ["_".join(pair) for pair in it.product(driver_names_to_process, trips)]

# On divise en nb process
driver_ids_batches = np.array_split(driver_ids_to_process, nb_process)
ids_batches = np.array_split(ids, nb_process)

for k in range(nb_process):
    Process(target=process_loop, args=(driver_ids_batches[k], ids_batches[k], nb_process, k)).start()

print 'C\'est parti !'

# SINGLE PROCESS

# trip_pointers = read_trip_pointers()
# Y_preds = []
# driver_ids_to_process, driver_names_to_process = sk.utils.shuffle(range(nd), drivers)
# ids = ["_".join(pair) for pair in it.product(driver_names_to_process, trips)]

# n_batches = nd/batch_size
# driver_ids_batches = np.array_split(driver_ids_to_process, n_batches)
# j = 1
# for i, driver_ids_batch in enumerate(driver_ids_batches):
#     logging.info("Reading batch %d of %d" % (i+1, n_batches))
#     driver_arrays = read_from_pickles(driver_ids_batch)

#     for d in driver_ids_batch:
#         logging.info("Processing driver %d (%d of %d)" % (d, j, nd))
#         try:
#             Y_pred = train_predict_driver(d, driver_ids_batch)
#         except Exception as e:
#             logging.exception('Failed !')
#             Y_pred = -np.ones(nt)
#         Y_preds += Y_pred.tolist()
#         j += 1

# result = np.array([ids, Y_preds]).T
# np.savetxt('submission_constance_et_nico1.csv', result, header="driver_trip,prob", comments="", fmt="%s", delimiter=",")

#####################################################
#   Running train / test for several drivers    #####
#####################################################

# MULTI PROCESS

# def f(i, d, driver_ids_batch, driver_arrays):
#     logging.info("Processing driver #%d (%d of %d)" % (d, i+1, nd_test))
#     Ys_true, Y_preds = train_test_driver(d, driver_ids_batch, driver_arrays)
#     for k in range(n_inits_test):
#         Y_true[k] += Ys_true[k].tolist()
#         Y_pred[k] += Y_preds[k].tolist()
#     auc = [sk.metrics.roc_auc_score(Yt, Yp) for Yt, Yp in zip(Y_true, Y_pred)]
#     result_queue.put(auc)

# trip_pointers = read_trip_pointers()
# Y_true = [[] for j in range(n_inits_test)]
# Y_pred = [[] for j in range(n_inits_test)]

# driver_ids_batch = [[], [], [], [], [], [], [], []]

# for i in range(nb_process):
#     driver_ids_batch[i] = random.sample(range(nd), nd_test)

# driver_ids_batch_all = np.array([driver_ids_batch[0], driver_ids_batch[1], driver_ids_batch[2], driver_ids_batch[3]])
# driver_arrays = read_from_pickles(driver_ids_batch[0]+driver_ids_batch[1]+driver_ids_batch[2]+driver_ids_batch[3])

# for i, d in enumerate(driver_ids_batch_all.T):
#     result_queue = Queue()
#     for j in range(nb_process):
#         Process(target=f, args=(i, d[j], driver_ids_batch_all[j], driver_arrays)).start()
#         #auc = f(i,d[j],driver_ids_batch_all[j], driver_arrays)
#     auc = result_queue.get() + result_queue.get() + result_queue.get()
#     logging.info("Mean AUC so far = %0.3f" % np.mean(auc))

# logging.info("AUCs: %s" % auc)
# logging.info("Mean AUC = %0.3f" % np.mean(auc))

# SINGLE PROCESS

# trip_pointers = read_trip_pointers()
# Y_true = [[] for j in range(n_inits_test)]
# Y_pred = [[] for j in range(n_inits_test)]
# driver_ids_batch = random.sample(range(nd), nd_test)
# driver_arrays = read_from_pickles(driver_ids_batch)

# for i, d in enumerate(driver_ids_batch):
#     logging.info("Processing driver #%d (%d of %d)" % (d, i+1, nd_test))
#     Ys_true, Y_preds = train_test_driver(d, driver_ids_batch, driver_arrays)
#     for k in range(n_inits_test):
#         Y_true[k] += Ys_true[k].tolist()
#         Y_pred[k] += Y_preds[k].tolist()

#     auc = [sk.metrics.roc_auc_score(Yt, Yp) for Yt, Yp in zip(Y_true, Y_pred)]
#     logging.info("Mean AUC so far = %0.3f" % np.mean(auc))

# logging.info("AUCs: %s" % auc)
# logging.info("Mean AUC = %0.3f" % np.mean(auc))

# #########################################################
# ##           Manual testing / feature creation        ###
# #########################################################

# d = 2453
# t = 81

# mean_roots_0 = []
# mean_roots_1 = []

# trip_pointers = read_trip_pointers()
# driver_arrays = read_from_pickles([d])
# trip = get_raw_trip(d, t, driver_arrays)

# feat = compute_features(trip, d_id=d, t_id=t)
# print feat


# x_orig, y_orig = trip[:, 0], trip[:, 1]
# mask = ((np.diff(trip[:, 0]) != 0) & (np.diff(trip[:, 1]) != 0))
# x_orig, y_orig = x_orig[mask], y_orig[mask]
# trip_interpol, u = sp.interpolate.splprep([x_orig, y_orig])
# u_grid = np.linspace(0, 1, 100)
# x_interpol, y_interpol = sp.interpolate.splev(u_grid, trip_interpol)
# mean_roots_0 += sp.interpolate.sproot(trip_interpol)[0].tolist()
# mean_roots_1 += sp.interpolate.sproot(trip_interpol)[1].tolist()


# speeds, accel = speeds_and_accelerations(trip)
# speeds *= 3.6
# high_accel = np.where(np.abs(accel) > 1.)

# plt.figure()
# plt.title("Interpolated")
# plt.plot(x_interpol, y_interpol, '.-', label='interpolated')
# for k in range(0, len(x_interpol), 10):
#     plt.annotate(str(k), xy=(x_interpol[k], y_interpol[k]))

# plt.figure()
# plt.title('Speed')
# plt.plot(speeds, '.-')
# plt.plot(high_accel, 2, '.', color='red')

# plt.show()
