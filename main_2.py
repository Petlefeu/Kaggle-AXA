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
logging.basicConfig(
	level=logging.INFO,
	format='[%(asctime)s][%(levelname)s] %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S',
	filename='kaggle.log'
)

nt = 200								# Number of trips per driver
nf = 86									# Number of features
csv_dir = "drivers"
pickles_dir = "pickles"
min_false_trips = 30					# False trips (from other drivers) we add to the correct ones (from the driver of iterest)
max_false_trips = 50					
n_noise = 10 							# Number of false trips we label as 'correct' for the training
n_correct_trips = 200					# Number of correct trips in each dataset

n_inits_train = 15						# Number of random datasets buit for train
n_inits_test = 5						# Number of random datasets buit for test

batch_size = 300 						# Number of drivers to process in one batch

nd_test = 200							# Number of drivers to use for AUC computation
drivers = os.listdir('drivers')[1:]		# List of driver names ('1', '10', '100', ...)
nd = len(drivers)
trips = map(str, range(1,nt+1))			# List of trip names (for one driver) ('1', '2', '3', ..., '200')


# ######################################
# #####            Utils           #####
# ######################################

percentile = lambda x,k: np.percentile(x, k) if x is not None and x.size else 0
mean = lambda x: np.mean(x) if x is not None and x.size else 0

def moving_average(a, n=3):
	if len(a.shape)==1:
		ret = np.cumsum(a, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		mvavg = ret[n - 1:] / n
	else:
		ret = np.cumsum(a, dtype=float, axis=0)
		ret[n:,:] = ret[n:,:] - ret[:-n,:]
		mvavg = ret[n - 1:,:] / n		
	return mvavg

# ######################################
# ## Pre-loading data into pickles  ####
# ######################################

def preload_into_pickles():
	trip_pointers = []
	for i,d in enumerate(drivers):
		logging.info("Loading driver %d of %d into pickle." % (i,nd))
		driver_trips = [np.genfromtxt("%s/%s/%s.csv" % (csv_dir,d,t), skip_header=True, delimiter=',') for t in trips]
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
	for i,d in enumerate(drivers_batch):
		logging.debug("Reading driver %d (#%s) from pickle (%d of %d)." % (d, drivers[d], i, N))
		array = np.load("%s/driver-%s.npy" % (pickles_dir, drivers[d]))
		driver_arrays[d] = array
	return driver_arrays

# ######################################
# #####     Features computation   #####
# ######################################
def compute_features(trip):
	# Removing stationary parts from trip
	clean_trip, is_moving_booleans = remove_stationary_points(trip)
	
	if len(clean_trip)>3:
		# Interpolating and evaluating
		u_grid = np.linspace(0,1,100)
		trip_spline, u_orig = interpolate(clean_trip)
		trip_regular, trip_regular_diff1, trip_regular_diff2 = evaluate_curves(trip_spline, u_grid) 
		trip_dynamic, trip_dynamic_diff1, trip_dynamic_diff2 = evaluate_curves(trip_spline, u_orig)

		# Getting feature vectors
		speeds, accelerations = get_speeds_and_accelerations(trip_dynamic_diff1, trip_dynamic_diff2, u_orig)
		headings = get_headings(trip_regular)

		curvature = get_curvature(trip_dynamic_diff1, trip_dynamic_diff2)
		centrifugal_accelerations = np.power(speeds, 2)*curvature

		# Some simple features
		trip_duration = len(trip)
		trip_length = np.sum(speeds)
		in_motion_ratio = np.mean(is_moving_booleans)

		features = [
			[trip_length],
			[trip_duration],
			[in_motion_ratio],
			[np.mean(speeds)],
			[np.mean(accelerations)],
			[np.mean(headings)],
			np.histogram(speeds, bins=[0,5,20,30,40,50,60,70,80,90,100,110,130,150,170,200], density=True)[0],
			np.histogram(accelerations, bins=[-10,-5,-2,-1.5,-1,-.8,-.6,-.4,-.2,-.15,-.1,-1e-2,-1e-3,-1e-4,0,1e-4,1e-3,1e-2,.1,.15,.2,.4,.6,.8,1,1.5,2,5,10], density=True)[0],
			np.histogram(headings, bins=[0,20,40,60,80,100,120,140,160,180], density=True)[0],
			np.histogram(curvature, bins=[0,1e-6,1e-5,1e-4,1e-3,1e-2,.1,.2,.4,.6,.8,1,2,3,4,5,6], density=True)[0],
			np.histogram(centrifugal_accelerations, bins=[0,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,.1,.2,.5,1,2,5], density=True)[0],
		]
	else:
		features = [
			[0],
			[len(trip)],
		]+[[0]]*(nf-2)
	return features


def get_raw_speeds_and_accelerations(trip):
	speed_vectors = np.diff(trip, axis=0)
	speeds = np.linalg.norm(speed_vectors, axis=1)*3.6
	accelerations = np.diff(speeds, axis=0)
	return speeds, accelerations

def get_speeds_and_accelerations(trip_dynamic_diff1, trip_dynamic_diff2, u_orig):
	speeds = np.linalg.norm(trip_dynamic_diff1, axis=1)
	accelerations = (trip_dynamic_diff1[:,0]*trip_dynamic_diff2[:,0]+trip_dynamic_diff1[:,1]*trip_dynamic_diff2[:,1])/speeds
	return 3.6*speeds, accelerations[np.isfinite(accelerations)]

def remove_stationary_points(trip):
	x, y = trip[:,0], trip[:,1]
	mask = ((np.diff(x) != 0) | (np.diff(y) != 0))
	x_clean, y_clean = x[mask], y[mask]
	trip_clean = np.vstack([x_clean, y_clean]).T
	return trip_clean, mask

# Interpolate the trip with a spline (3rd degree piecewise polynomial functions)
# and returns the spline object (as outputted by splprep) and the original time u
def interpolate(trip):
	trip_spline, u_orig = sp.interpolate.splprep(trip.T, k=3)
	return trip_spline, u_orig

# Get differenciated curves (degree 0, 1 and 2) for the given time
def evaluate_curves(trip_spline, u):
	du = np.diff(u)
	x, y = sp.interpolate.splev(u, trip_spline)
	dx, dy = sp.interpolate.splev(u, trip_spline, der=1)
	d2x, d2y = sp.interpolate.splev(u, trip_spline, der=2)
	return (
		np.vstack([x, y]).T,
		np.vstack([du*dx[0:-1], du*dy[0:-1]]).T,
		np.vstack([du*du*d2x[0:-1], du*du*d2y[0:-1]]).T,
	)

# Return the curvature curve of the road
def get_curvature(trip_regular_diff1, trip_regular_diff2):
	dx, dy = trip_regular_diff1[:,0], trip_regular_diff1[:,1]
	d2x, d2y = trip_regular_diff2[:,0], trip_regular_diff2[:,1]
	curvature = np.abs(dx*d2y-dy*d2x)/np.power(np.power(dx, 2)+np.power(dy, 2), 1.5)
	return curvature[np.isfinite(curvature)]

# Returns the angle between the speed vector and the x axis (in degrees)
def get_headings(trip_regular):
	diffs = np.diff(trip_regular, axis=0)
	headings = np.arccos(diffs[:,0]/np.linalg.norm(diffs, axis=1))*180/np.pi
	return headings[np.isfinite(headings)]


# ######################################
# #####     Building data set      #####
# ######################################

# Returns the trip 't' of a driver 'd' (already converted into features)
# 'd' ranges from 0 to nd-1
# 't' ranges from 0 to nt-1
def get_trip(d, t):
	try:
		features_list = compute_features(get_raw_trip(d,t))
		features = np.hstack(features_list)
		not_finite_indices = ~np.isfinite(features)
		if sum(not_finite_indices):
			logging.warning('Trip %d of driver %d has some NaN of infinite features.' % (t, d))
			features[not_finite_indices] = 0
		return features
	except Exception as e:
		logging.exception('Cannot compute features for trip %d of driver %d (error).' % (t, d))
		return np.zeros(nf)

def get_raw_trip(d, t):
	t = int(t)
	begin = 0 if t==0 else trip_pointers[d,t-1]
	end = None if t==nt else trip_pointers[d,t]
	raw_trip = driver_arrays[d][begin:end,:]
	return raw_trip

# Builds a dataset for a specific driver (with some correct trips and some false ones, at random)
def build_driver_train_dataset(driver, correct_trips, drivers_batch):
	# Generating false trips
	n_false_trips = random.randint(min_false_trips, max_false_trips)
	logging.debug("Building dataset with %d correct trips and %d false ones" % (len(correct_trips),n_false_trips))
	false_drivers = random.sample(drivers_batch, n_false_trips)
	false_trips = [get_trip(d, random.randint(0,nt-1)) for d in false_drivers]
	false_trips = [f for f in false_trips if sum(f)>0]
	n_false_trips = len(false_trips)

	# Building data
	X = np.array(correct_trips+false_trips)											# One row = 1 trip (each column is a feature)
	Y_true = np.concatenate([np.ones(len(correct_trips)),np.zeros(n_false_trips)])	# Target variable (1 if correct driver, 0 otherwise)
	Y_noisy = Y_true.copy()
	Y_noisy[n_correct_trips+np.random.randint(0,n_false_trips-1,n_noise)] = 1 		# Adding noise (false trips labelled as true)
	X, Y_noisy, Y_true = sk.utils.shuffle(X, Y_noisy, Y_true)

	return X, Y_noisy, Y_true

# ######################################
# #####     Training & testing     #####
# ######################################

# Trains a classifier for the given driver
def train(driver, correct_trips, drivers_batch):
	# Subsampling correct_trips
	correct_trips = random.sample(correct_trips, n_correct_trips)

	# Getting dataset
	X, Y_noisy, _ = build_driver_train_dataset(driver, correct_trips, drivers_batch)

	# Training
	# cls = sk.linear_model.LogisticRegression(C=1000.0)
	# cls = sk.svm.SVC(C=10, probability=True)
	cls =  sk.ensemble.RandomForestClassifier(n_estimators=20)
	cls.fit(X, Y_noisy)

	return cls


def train_test_driver(d, driver_ids_batch):
	# Building correct trips
	correct_trips = [get_trip(d, t) for t in range(nt)]

	# Getting classifiers
	classifiers = [train(d, correct_trips, driver_ids_batch) for i in range(n_inits_train)]

	# Predicting on new random datasets
	datasets = [build_driver_train_dataset(d, correct_trips, driver_ids_batch) for i in range(n_inits_test)]
	Xs = [d[0] for d in datasets]
	Ys_true = [d[2] for d in datasets]
	Y_preds = [np.array([cls.predict_proba(X)[:,1] for cls in classifiers]) for X in Xs]
	Y_preds_aggreg = [ys.mean(0) for ys in Y_preds]
	
	return Ys_true,Y_preds_aggreg


def train_predict_driver(d, drivers_batch):
	# Building correct trips
	correct_trips = [get_trip(d,t) for t in range(nt)]

	# Getting classifiers
	classifiers = [train(d, correct_trips, drivers_batch) for i in range(n_inits_train)]

	# Predicting on all the driver's trips
	X = np.array(correct_trips)
	Y_preds = np.array([cls.predict_proba(X)[:,1] for cls in classifiers])
	Y_preds_aggreg = Y_preds.mean(0)

	return Y_preds_aggreg


# #########################################################
# #####  			Pre-oading data 			   	   #####
# #########################################################
# preload_into_pickles()


# #########################################################
# #####   Running train / test for several drivers    #####
# #########################################################

# trip_pointers = read_trip_pointers()
# Y_true = [[] for j in range(n_inits_test)]
# Y_pred = [[] for j in range(n_inits_test)]
# driver_ids_batch = random.sample(range(nd), nd_test)
# driver_arrays = read_from_pickles(driver_ids_batch)

# for i,d in enumerate(driver_ids_batch):
# 	logging.info("Processing driver #%d (%d of %d)" % (d,i+1,nd_test))
# 	Ys_true,Y_preds = train_test_driver(d, driver_ids_batch)
# 	for k in range(n_inits_test):
# 		Y_true[k] += Ys_true[k].tolist()
# 		Y_pred[k] += Y_preds[k].tolist()

# 	auc = [sk.metrics.roc_auc_score(Yt, Yp) for Yt,Yp in zip(Y_true,Y_pred)]
# 	logging.info("Mean AUC so far = %0.3f" % np.mean(auc))

# logging.info("AUCs: %s" % auc)
# logging.info("Mean AUC = %0.3f" % np.mean(auc))


# #########################################################
# ##    Computing final predictions for all drivers     ###
# #########################################################

trip_pointers = read_trip_pointers()
Y_preds = []
driver_ids_to_process, driver_names_to_process = sk.utils.shuffle(range(nd), drivers)
ids = ["_".join(pair) for pair in it.product(driver_names_to_process, trips)]

n_batches = nd/batch_size
driver_ids_batches = np.array_split(driver_ids_to_process, n_batches)
j = 1
for i,driver_ids_batch in enumerate(driver_ids_batches):
	# driver_ids_batch = [1342, 1343]+driver_ids_batch.tolist()
	logging.info("Reading batch %d of %d" % (i+1,n_batches))
	driver_arrays = read_from_pickles(driver_ids_batch)

	for d in driver_ids_batch:
		logging.info("Processing driver %d (%d of %d)" % (d,j,nd))
		try:
			Y_pred = train_predict_driver(d, driver_ids_batch)
		except Exception as e:
			logging.exception('Failed !')
			Y_pred = -np.ones(nt)
		Y_preds += Y_pred.tolist()
		j+=1

result = np.array([ids, Y_preds]).T
np.savetxt('submission_allinterpol_2_v4.6.csv', result, header="driver_trip,prob", comments="", fmt="%s", delimiter=",")

