"""
Perceptual decision-making task, loosely based on the random dot motion
discrimination task.

  Response of neurons in the lateral intraparietal area during a combined visual
  discrimination reaction time task.
  J. D. Roitman & M. N. Shadlen, JNS 2002.

  http://www.jneurosci.org/content/22/21/9475.abstract

Reaction-time version.

"""
from __future__ import division

import numpy as np

from pycog import tasktools

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin  = 3
N    = 100
Nout = 2

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Start cue
START = 2

#-----------------------------------------------------------------------------------------
# Output connectivity
#-----------------------------------------------------------------------------------------

Cout = np.zeros((Nout, N))
Cout[:,EXC] = 1

#-----------------------------------------------------------------------------------------
# Task structure
#-----------------------------------------------------------------------------------------

cohs        = [1, 2, 4, 8, 16]              # coherences
in_outs     = [1, -1]                       # in (1) is the value of one output unit target values which is in the receptive field of that neuron (the output unit)
nconditions = len(cohs)*len(in_outs)        # 10 different task to train 
pcatch      = 5/(nconditions + 1)           # wtf????????????????????

SCALE = 3.2
def scale(coh):
    return (1 + SCALE*coh/100)/2

def generate_trial(rng, dt, params):      # is called in dataset.Dataset.update to generate trial and put them in u and target (inputs = [u, targets]) detailed in trainer
    #-------------------------------------------------------------------------------------
    # Select task condition
    #-------------------------------------------------------------------------------------

    catch_trial = False                             # when catch_trial is on all the inputs are zero --> train the network not to make any decision
    if params['name'] in ['gradient', 'test']:     # the name of the Dataset object: now we have 'gradient' or 'validation' or default: 'Dataset'
        if params.get('catch', rng.rand() < pcatch):   # if there's no 'catch' in the dict get the boolean rng.rand() < pcatch instead, now pcatch is 5/11
            catch_trial = True
        else:
            coh    = params.get('coh',    rng.choice(cohs))         # return random value from cohs
            in_out = params.get('in_out', rng.choice(in_outs))      # same
    elif params['name'] == 'validation':            # particularly for validation dataset
        b = params['minibatch_index'] % (nconditions + 1)           # 'minibatch_index' is only in the update method usually 20 for n_gradient and 1000 for n_validation, b = 10
        if b == 0:
            catch_trial = True
        else:
            k0, k1 = tasktools.unravel_index(b-1, (len(cohs), len(in_outs)))  # k0 = 4, k1 = 1
            coh    = cohs[k0]       # 8, obviously
            in_out = in_outs[k1]    # -1                ===>  catch_trial = False
    else:
        raise ValueError("Unknown trial type.")

    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    if catch_trial:
        epochs = {'T': 2000}
    else:
        if params['name'] == 'test':                # there's no 'test' in params['name'] in whatever cases
            fixation = 300
            stimulus = 1500
        else:
            fixation = 100
            stimulus = 800
        no_reward = 300
        T         = fixation + stimulus

        epochs = {
            'fixation': (0, fixation),
            'stimulus': (fixation, T),
            'decision': (fixation + no_reward, T)
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
        if in_out > 0:
            choice = 0
        else:
            choice = 1

        # Trial info
        trial['info'] = {'coh': coh, 'in_out': in_out, 'choice': choice}

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    X = np.zeros((len(t), Nin))
    if not catch_trial:
        # Stimulus
        X[e['stimulus'],choice]   = scale(+coh)
        X[e['stimulus'],1-choice] = scale(-coh)

        # Start cue
        X[e['stimulus'],START] = 1
    trial['inputs'] = X

    #-------------------------------------------------------------------------------------
    # Target output
    #-------------------------------------------------------------------------------------

    if params.get('target_output', False):                  #check if params['target_output'] exist because we will always assign it to be True
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.zeros_like(Y)         # Mask matrix

        # Hold values
        hi = 1.2
        lo = 0.2

        if catch_trial:
            Y[:] = lo
            M[:] = 1
        else:
            # Fixation
            Y[e['fixation'],:] = lo

            # Decision
            Y[e['decision'],choice]   = hi
            Y[e['decision'],1-choice] = lo

            # Only care about fixation and decision periods
            M[e['fixation']+e['decision'],:] = 1

        # Outputs and mask
        trial['outputs'] = Y
        trial['mask']    = M

    #-------------------------------------------------------------------------------------

    return trial

# Performance measure
performance = tasktools.performance_2afc

# Termination criterion
TARGET_PERFORMANCE = 85
def terminate(pcorrect_history):
    return np.mean(pcorrect_history[-5:]) > TARGET_PERFORMANCE

# Validation dataset
n_validation = 100*(nconditions + 1)
