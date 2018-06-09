#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 18:43:53 2018
Model fronto-parietal network performing sequential category-based decisions
@Aiden_Bhe

  Computing by robust transience: How the fronto-parietal network performs sequential category-based decisions.
  Warasinee Chaisangmongkon, Sruthi K. Swaminathan, David J. Freedman, and Xiao-Jing Wang, Neuron. 2017 March 22;
  93(6): 1504–1517.e4. doi:10.1016/j.neuron.2017.03.002.

Using pycog https://github.com/frsong/pycog

"""
from __future__ import division

import numpy as np

from pycog import tasktools
from pycog.connectivity import Connectivity

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 33
N    = 150
Nout = 6

# Noises
var_in  = 0.01**2
var_rec = 0.01**2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)
Nexc = len(EXC)
Ninh = len(INH)

# Time step size
dt = 10 # ms

# Populations
EXC_IN  = EXC[:Nexc//4] # because in this model the input units are sparsely connected to all excitatory recurrent units
EXC_OUT = EXC[Nexc//4:Nexc//2]  # 30 recurrent units that connected to action units [0] and [1]
EXC_OUT_0 = EXC[Nexc//4:3*Nexc//8] # 15 recurrent units that connected to action units [0]
EXC_OUT_1 = EXC[3*Nexc//8:Nexc//2] # 15 recurrent units that connected to action units [1]
EXC_NS  = EXC[Nexc//2:]

#-----------------------------------------------------------------------------------------
# Input connectivity
#-----------------------------------------------------------------------------------------

Cin = np.ones((N,Nin))
#Cin = np.zeros((N, Nin))
#for i in EXC_IN:
#    for j in range(0,Nin):
#        Cin[i,j] = 1

#-----------------------------------------------------------------------------------------
# Recurrent connectivity
#-----------------------------------------------------------------------------------------

Crec = np.zeros((N, N))
for i in EXC_IN:
    Crec[i,EXC_IN]  = 1
    Crec[i,i]       = 0
    Crec[i,EXC_OUT] = 1 # mute this line to constrain
    Crec[i,EXC_NS]  = 1
    Crec[i,INH]     = np.sum(Crec[i,EXC])/len(INH)
for i in EXC_OUT:
    Crec[i,EXC_IN]  = 1 # mute this line to constrain
    Crec[i,EXC_OUT] = 1
    Crec[i,i]       = 0
    Crec[i,EXC_NS]  = 1
    Crec[i,INH]     = np.sum(Crec[i,EXC])/len(INH)
for i in EXC_NS:
    Crec[i,EXC]     = 1
    Crec[i,i]       = 0
    Crec[i,INH]     = np.sum(Crec[i,EXC])/len(INH)
for i in INH:
    Crec[i,EXC]     = 1
    Crec[i,INH]     = np.sum(Crec[i,EXC])/(len(INH)-1)
    Crec[i,i]       = 0
Crec /= np.linalg.norm(Crec, axis=1)[:,np.newaxis]      # normalize

#-----------------------------------------------------------------------------------------
# Output connectivity
#-----------------------------------------------------------------------------------------

"""
     All synapses connected to Action Units are plastic 
     while the control units are non-plastic
     
"""
#Action units connectivity
Cout_plastic = np.zeros((Nout, N))
Cout_plastic[0,EXC_OUT_0] = 1
Cout_plastic[1,EXC_OUT_1] = 1

#Control units connectivity
Cout_fixed = np.zeros((Nout, N))
Cout_fixed[2,EXC_OUT_0] = 1    # this unit compute the sum of activities of the 15 units
Cout_fixed[3,EXC_OUT_1] = 1    # this unit compute the sum of activities of the 15 units
Cout_fixed[4,EXC] = 1 # this unit compute the sum of all excitatory activities
Cout_fixed[5,INH] = 1 # sum of all inhibitory activities

Cout = Connectivity(Cout_plastic,Cout_fixed)

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

#phi = list(np.arange(15,180,30))
phi = [90]
in_outs     = [1, -1]
matches     = [1, -1]
nconditions = len(in_outs)*len(matches)*len(phi)**2
pcatch      = 1/(nconditions + 1)

def generate_trial(rng, dt, params):      # is called in dataset.Dataset.update to generate trial and put them in u and target (inputs = [u, targets]) detailed in trainer
    #-------------------------------------------------------------------------------------
    # Select task condition
    #-------------------------------------------------------------------------------------

    angle_s = 15
    angle_t = 15
    match = 0
    catch_trial = False                             # when catch_trial is on all the inputs are zero --> train the network not to make any decision
    if params['name'] in ['gradient', 'test']:     # the name of the Dataset object: now we have 'gradient' or 'validation' or default: 'Dataset'
        if params.get('catch', rng.rand() < pcatch):   # if there's no 'catch' in the dict get the boolean rng.rand() < pcatch instead, now pcatch is 5/11
            catch_trial = True
        else:
            in_out   = rng.choice(in_outs)
            angle_s  = params.get('sample angle', in_out*rng.choice(phi))         # return random value from phi
            match    = rng.choice(matches)
            angle_t  = params.get('test angle', match*in_out*rng.choice(phi))
    elif params['name'] == 'validation':            # particularly for validation dataset
        b = params['minibatch_index'] % (nconditions + 1)           # 'minibatch_index' is looped from 0 until batch_size
        if b == 0:                                                  # which means if validation batch size = 1000 b will starts from 0 to 144 (nConditions)
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b-1, (2*len(phi), 2*len(phi)))  
            phi_full  = phi + list(-np.asarray(phi))
            phi_full  = np.sort(phi_full)
            angle_s   = phi_full[k0]
            angle_t   = phi_full[k1]    
    else:
        raise ValueError("Unknown trial type.")    
    
    if angle_s/angle_t < 0:
        match = -1  
    elif angle_s/angle_t > 0:
        match = 1
    
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 4400}
    else:
        resting  = 1000
        fixation = 500
        sample   = 650
        delay    = 1000
        test     = 250
        choice   = 1000
        T        = resting + fixation + sample + delay + test + choice

        epochs = {
            'resting' : (0,resting),
            'fixation': (resting, resting + fixation),
            'sample'  : (resting + fixation, resting + fixation + sample),
            'delay'   : (resting + fixation + sample, resting + fixation + sample + delay),
            'test'    : (resting + fixation + sample + delay, resting + fixation + sample + delay + test),
            'choice'  : (resting + fixation + sample + delay + test, T)
            }
        epochs['T'] = T

    #-------------------------------------------------------------------------------------
    # Trial info
    #-------------------------------------------------------------------------------------

    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    if catch_trial:
        trial['info'] = {}
    else:
        # Correct choice
        if match > 0:
            choice = 0
        else:
            if match == 0:
                raise ValueError("wrong choice.") 
            choice = 1

        # Trial info
        trial['info'] = {'sample angle': angle_s, 'test angle': angle_t, 'choice': choice}

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    zigma = 43.2 # degree
    a     = 0.8
    theta = np.linspace(-168.75, 180, 32)    
    
    U = np.zeros((len(t), Nin))
    if not catch_trial:
        # Stimulus
        for units in range(0, Nin-1):
            if not abs(angle_s - theta[units]) > 180: 
                encoded_sample = a * np.exp(-((angle_s - theta[units])**2)/(2*(zigma**2)))# Gaussian tuning curve
            else:
                encoded_sample = a * np.exp(-((360-abs(angle_s - theta[units]))**2)/(2*(zigma**2)))
            if not abs(angle_t - theta[units]) > 180: 
                encoded_test = a * np.exp(-((angle_t - theta[units])**2)/(2*(zigma**2)))# Gaussian tuning curve
            else:
                encoded_test = a * np.exp(-((360-abs(angle_t - theta[units]))**2)/(2*(zigma**2)))
            U[e['sample'], units] = encoded_sample
            U[e['test'], units]   = encoded_test
        U[e['fixation'], Nin-1]   = 0.05                    # signals the appearance of fixation dot 
    
    trial['inputs'] = U

    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    if params.get('target_output', False):                  #check if params['target_output'] exist because in this case we always assign it to be True
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.zeros_like(Y)         # Mask matrix

        # Hold values
        match_choice        = 5.0
        match_control       = 2.0
        rest_control_exc    = 2.0
        rest_control_inh    = -0.5

        if catch_trial:
            # Stay at Resting
            Y[:, 4] = rest_control_exc
            Y[:, 5] = rest_control_inh
            for units in [0, 1, 4, 5]:
                M[:, units]  = 1
        else:
            # Resting
            Y[e['resting'], 4] = rest_control_exc
            Y[e['resting'], 5] = rest_control_inh
            for units in [0, 1, 4, 5]:
                M[e['resting'],units] = 1
            
            # From fixation to the end of delay
            for units in range(2):
                M[e['fixation']+e['sample']+e['delay'], units] = 1
            
            # Choice
            Y[e['choice'], choice]   = match_choice
            Y[e['choice'], 2 + choice]   = match_control
            for units in range(0,4):
                M[e['choice'], units] = 1

        # Outputs and mask
        trial['outputs'] = Y
        trial['mask']    = M

    #-------------------------------------------------------------------------------------

    return trial

# Performance measure
performance = tasktools.performance_2afc

# Termination criterion
TARGET_PERFORMANCE = 88.76  #88.76
def terminate(pcorrect_history):
    return np.mean(pcorrect_history[-5:]) > TARGET_PERFORMANCE

# Validation dataset
#n_validation = 10*(nconditions + 1)
#n_gradient   = 200