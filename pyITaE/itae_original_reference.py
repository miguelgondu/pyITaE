#! /usr/bin/env python
#| This file is a part of the pyite framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#| Antoine Cully, Jeff Clune, Danesh Tarapore, and Jean-Baptiste Mouret.
#|"Robots that can adapt like animals." Nature 521, no. 7553 (2015): 503-507.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
from ite_utils import *
import sys
import time
sys.path.append('..')

class ITE:
    def __init__(self):
        pass

def ite(params):
	# Load the map
	centroids = load_centroids(sys.argv[1])
	fits, descs, ctrls = load_data(sys.argv[2], centroids.shape[1],dim_ctrl)

	# Filtering, which doesn't actually take place
	n_fits, n_descs, n_ctrls = [], [], []
	for i in range(0,len(fits)):
		#if(fits[i]>1): #filter the fitness , discard the ones < 1
		if(1):
			n_fits.append(fits[i])
			n_descs.append(descs[i])
			n_ctrls.append(ctrls[i])
	n_fits = np.array(n_fits)
	n_descs = np.array(n_descs)
	n_ctrls = np.array(n_ctrls)

	# MIGUEL:
	# the n_whatever are essentially copies of the original fits but as numpy arrays.

	# Huh, they copy the "performance map" twice.
	# do they modify n_fits?, what's the need for copying twice.
	# They don't. Funny. They work with n_fits_real
	# and fits_saved from here on.
	n_fits_real = copy(np.array(n_fits))
	fits_saved = copy(n_fits)
	next_index_to_test = np.argmax(n_fits) # The first cell to test is the best fitness, yeah.

	################ INIT ITE  ###########################################################
	robot_utils.end_episode_distance = -1000 
	prev_ep_distance = -1000

	init, run = True, True
	num_it = 0
	# QUESTION: why do they start the real preferences like this?, just as a "step -1" or something?
	# ANSWER: yes, this is an initialization, as it says above.
	# Q: why do they instantiate real_perfs here if they overwrite it on the init step.
	real_perfs, tested_indexes, tested_ctrls, tested_descs =[-1000], [], [], []
	X, Y = [], []
	############## BEGIN ITE ###################################################
	while(run):
		stop_cond =  alpha*max(n_fits_real)

		if(not init):
			#define GP kernerl
			ker = GPy.kern.Matern52(dim_x,lengthscale = rho, ARD=False) + GPy.kern.White(dim_x,np.sqrt(variance_noise_square))
			#define Gp which is here the difference btwn map perf and real perf
			m = GPy.models.GPRegression(X,Y,ker)
			#predict means and variances for the difference btwn map perf and real perf
			means, variances = m.predict(n_descs)
			# Add the predicted difference to the map found in simulation
			# They overwrite the whole n_fits_real!,
			for j in range(0,len(n_fits_real)):
				n_fits_real[j] = means[j] + fits_saved[j]
			#Compute how many percent of the behaviors have been significantly changed by this iteration
			changed = 0
			if(len(means_prev)>0):
				for y in range(0,len(means)):
					if(abs(means[y]-means_prev[y])>0.1):
						changed = changed + 1
				percent = changed*100/len(means)
				percents.append(percent)
			means_prev = means
			#apply acquisition function to get next index to test
			next_index_to_test = UCB(n_fits_real,kappa,variances)
		else:
			real_perfs = [] # Q: Why do they overwrite the [-1000].
			init = False
		#if the behavior to test has already been tested, don't test it again
		if(next_index_to_test in tested_indexes):
			# What's happening here?!, an np.index wouldn't suffice?
			for h in range(0,len(tested_indexes)):
				if(next_index_to_test==tested_indexes[h]):
					tmp_id = h
			real_perf = real_perfs[tmp_id]
			ctrl_to_test = n_ctrls[next_index_to_test]
			tested_indexes.append(next_index_to_test)
			in_count = in_count + 1
		else:
			# Run the simulation and append the index
			# to the set of tested indexes.
			ctrl_to_test = n_ctrls[next_index_to_test]
			tested_indexes.append(next_index_to_test)
			# eval the real performance
			if(ros==1):
				eval_ros(episode_duration, ctrl_to_test, rate,ros)
				if(robot_utils.end_episode_distance == prev_ep_distance):
					sys.exit("WARNING : ite hasn't been able to recover the latest distance, exiting ite")
				else:
					prev_ep_distance = robot_utils.end_episode_distance
					real_perf = robot_utils.end_episode_distance
			else:
				real_perf, real_desc = eval_sim(ctrl_to_test, gui=params["gui"])
		if(len(X)==0):
			X.append(n_descs[next_index_to_test])
			Y.append(np.array(real_perf) - fits_saved[next_index_to_test]) # THIS is the important part.
			X = np.array(X)
			Y = np.array(Y)
		else:
			# Else they just stack (maintain) the vector, add more stuff, like Danny does.
			X = np.vstack((X, n_descs[next_index_to_test]))
			Y = np.vstack((Y, np.array(real_perf)-fits_saved[next_index_to_test]))
		
		# store it
		real_perfs.append(real_perf)
		tested_ctrls.append(ctrl_to_test)
		# store the tested descs
		tested_descs.append(n_descs[next_index_to_test])
		o = np.argmax(real_perfs)
		max_index = tested_indexes[o]
		max_perf = real_perfs[o]
		print("Max performance found with ",num_it," iterations : ", max(real_perfs))
		print("Associated max index ", max_index)
		print(real_perfs)
		print(tested_indexes)
		num_it = num_it + 1
		cond = max(real_perfs)
		if(cond > stop_cond):
			run = False
		if(num_it == max_iteration_number):
			run = False

	print(" ")
	print("Max performance found with ",num_it," iterations : ", max(real_perfs))
	print("Associated max index ", max_index)
	print("Associated controller ",tested_ctrls[np.argmax(real_perfs)])
	print("Associated descriptor ", tested_descs[np.argmax(real_perfs)])
	best_ctrls.append(tested_ctrls[np.argmax(real_perfs)])
	best_descs.append(tested_descs[np.argmax(real_perfs)])
	num_its.append(num_it)
	percent_affected_in_map.append(np.mean(np.array(percents)))
	in_number.append(in_count)
	mean_final.append(means)
	var_final.append(variances)
	best_dist.append(max(real_perfs))
	best_index.append(max_index)
	tested_descs_maps.append(tested_descs)
	tested_ctrls_maps.append(tested_ctrls)
	tested_indexes_maps.append(tested_indexes)
	tested_perfs_maps.append(real_perfs)
	np.save('num_its',np.array(num_its))
	np.save('percent_affected_in_map',np.array(percent_affected_in_map))
	np.save('in_number',np.array(in_number))
	np.save('mean_final',np.array(mean_final))
	np.save('var_final',np.array(var_final))
	np.save('best_dist',np.array(best_dist))
	np.save('best_index',np.array(best_index))
	np.save('best_ctrls',np.array(best_ctrls))
	np.save('best_descs',np.array(best_descs))
	np.save('tested_descs_maps',np.array(tested_descs_maps))
	np.save('tested_ctrls_maps',np.array(tested_ctrls_maps))
	np.save('tested_perfs_maps',np.array(tested_perfs_maps))
	np.save('tested_indexes_maps',np.array(tested_indexes_maps))
	# plot_GP(mu_map,sigma_map,fits,descs)

params = \
	    {
	        "pybullet_minitaur_sim_path": "pybullet_minitaur_sim",
			"pyhexapod_path": "pyhexapod",
			"gui" : True,
	    }

if __name__ == "__main__":
	ite(params)
