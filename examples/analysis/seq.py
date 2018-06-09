#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:37:10 2018

@author: aiden
"""

from __future__ import division

import pickle
import os
import sys
from   os.path import join

import numpy as np

from pycog          import fittools, RNN, tasktools
from pycog.figtools import Figure

THIS = "examples.analysis.seq"

#=========================================================================================
# Setup
#=========================================================================================

# File to store trials in
def get_trialsfile(p):
    return join(p['trialspath'], p['name'] + '_trials.pkl')

# Load trials
def load_trials(trialsfile):
    with open(trialsfile,'rb' ) as f:
        trials = pickle.load(f)

    return trials, len(trials)

# File to store sorted trials in
def get_sortedfile_stim_onset(p):
    return join(p['trialspath'], p['name'] + '_sorted_stim_onset.pkl')

# File to store sorted trials in
def get_sortedfile_response(p):
    return join(p['trialspath'], p['name'] + '_sorted_response.pkl')

# File to store d'
def get_dprimefile(p):
    return join(p['datapath'], p['name'] + '_dprime.txt')

# File to store selectivity
def get_selectivityfile(p):
    return join(p['datapath'], p['name'] + '_selectivity.txt')

def safe_divide(x):
    if x == 0:
        return 0
    return 1/x

# Define "active" units
def is_active(r):
    return np.std(r) > 0.1

# Nice colors to represent coherences, from http://colorbrewer2.org/
colors = {
        0:  '#c6dbef',
        1:  '#9ecae1',
        2:  '#6baed6',
        4:  '#4292c6',
        8:  '#2171b5',
        16: '#084594'
        }

# Decision threshold
THRESHOLD = 4.25

# Simple choice function
def get_choice(trial, threshold=True):
    if not threshold:
        return np.argmax(trial['z'][:2,-1])

    # Reaction time
    w0, = np.where(trial['z'][0] > THRESHOLD)
    w1, = np.where(trial['z'][1] > THRESHOLD)
    if len(w0) == 0 and len(w1) == 0:
        return None

    if len(w1) == 0:
        return 0, w0[0]
    if len(w0) == 0:
        return 1, w1[0]
    if w0[0] < w1[0]:
        return 0, w0[0]
    return 1, w1[0]

#=========================================================================================

def run_trials(p, args):
    """
    Run trials.

    """
    # Model
    m = p['model']

    # Number of trials
    try:
        ntrials = int(args[0])
    except:
        ntrials = 10
    ntrials *= m.nconditions + 1
   
    # RNN
    rng = np.random.RandomState(p['seed'])
    rnn = RNN(p['savefile'], {'dt': p['dt']}, verbose=False)

    # Trials
    w = len(str(ntrials))
    trials = []
    backspaces = 0
    try:
        for i in range(ntrials):
            
            # Trial
            trial_func = m.generate_trial
            trial_args = {
                'name':   'test',
                'catch':  False,
                }
            
            b = i % (m.nconditions + 1)
            angle_s = 0
            angle_t = 0
            if b == 0:
                # no sample and test given
                trial_args['catch'] = True
            else:
                # All other conditions
                k0, k1 = tasktools.unravel_index(b-1, (2*len(m.phi), 2*len(m.phi)))  
                phi_full  = m.phi + list(-np.asarray(m.phi))
                phi_full  = np.sort(phi_full)
                angle_s   = phi_full[k0]
                angle_t   = phi_full[k1]

            trial_args['sample angle'] = angle_s
            trial_args['test angle']   = angle_t
            info = rnn.run(inputs=(trial_func, trial_args), rng=rng)

            # Display trial type
            if b == 0:
                s = "Trial {:>{}}/{}: catch trial".format(i+1, w, ntrials)
            else:
                s = ("Trial {:>{}}/{}: {:>+3} -> {:>+3}"
                     .format(i+1, w, ntrials, angle_s, angle_t))
            sys.stdout.write(backspaces*'\b' + s)
            sys.stdout.flush()
            backspaces = len(s) + 1

            # Save
            dt    = rnn.t[1] - rnn.t[0]
            step  = int(p['dt_save']/dt)
            trial = {
                't':    rnn.t[::step],
                'u':    rnn.u[:,::step],
                'r':    rnn.r[:,::step],
                'z':    rnn.z[:,::step],
                'info': info
                }
            trials.append(trial)
    except KeyboardInterrupt:
        pass
    print("")

    # Save all
    filename = get_trialsfile(p)
    with open(filename, 'wb') as f:
        pickle.dump(trials, f, pickle.HIGHEST_PROTOCOL)
    size = os.path.getsize(filename)*1e-9
    print("[ {}.run_trials ] Trials saved to {} ({:.1f} GB)".format(THIS, filename, size))


#=========================================================================================

def do(action, args, p):
    """
    Manage tasks.

    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    #-------------------------------------------------------------------------------------
    # Trials
    #-------------------------------------------------------------------------------------

    if action == 'trials':
        run_trials(p, args)

    #-------------------------------------------------------------------------------------
    # Psychometric function
    #-------------------------------------------------------------------------------------

    elif action == 'psychometric':
        threshold = False
        if 'threshold' in args:
            threshold = True

        fig  = Figure()
        plot = fig.add()

        #---------------------------------------------------------------------------------
        # Plot
        #---------------------------------------------------------------------------------

        trialsfile = get_trialsfile(p)
        psychometric_function(trialsfile, plot, threshold=threshold)

        plot.xlabel(r'Percent coherence toward $T_\text{in}$')
        plot.ylabel(r'Percent $T_\text{in}$')

        #---------------------------------------------------------------------------------

        fig.save(path=p['figspath'], name=p['name']+'_'+action)
        fig.close()

    #-------------------------------------------------------------------------------------
    # Sort
    #-------------------------------------------------------------------------------------

    elif action == 'sort_stim_onset':
        sort_trials_stim_onset(get_trialsfile(p), get_sortedfile_stim_onset(p))

    elif action == 'sort_response':
        sort_trials_response(get_trialsfile(p), get_sortedfile_response(p))

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity aligned to stimulus onset
    #-------------------------------------------------------------------------------------

    elif action == 'units_stim_onset':
        from glob import glob

        # Remove existing files
        filenames = glob(join(p['figspath'], p['name'] + '_stim_onset_unit*'))
        for filename in filenames:
            os.remove(filename)
            print("Removed {}".format(filename))

        # Load sorted trials
        sortedfile = get_sortedfile_stim_onset(p)
        with open(sortedfile) as f:
            t, sorted_trials = pickle.load(f)

        for i in range(p['model'].N):
            # Check if the unit does anything
            active = False
            for r in sorted_trials.values():
                if is_active(r[i]):
                    active = True
                    break
            if not active:
                continue

            fig  = Figure()
            plot = fig.add()

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            props = {'prop': {'size': 8}, 'handletextpad': 1.02, 'labelspacing': 0.6}
            plot.legend(bbox_to_anchor=(0.18, 1), **props)

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'],
                     name=p['name']+'_stim_onset_unit{:03d}'.format(i))
            fig.close()

    #-------------------------------------------------------------------------------------
    # Plot single-unit activity aligned to response
    #-------------------------------------------------------------------------------------

    elif action == 'units_response':
        from glob import glob

        # Remove existing files
        filenames = glob(join(p['figspath'], p['name'] + '_response_unit*'))
        for filename in filenames:
            os.remove(filename)
            print("Removed {}".format(filename))

        # Load sorted trials
        sortedfile = get_sortedfile_response(p)
        with open(sortedfile) as f:
            t, sorted_trials = pickle.load(f)

        for i in range(p['model'].N):
            # Check if the unit does anything
            active = False
            for r in sorted_trials.values():
                if is_active(r[i]):
                    active = True
                    break
            if not active:
                continue

            fig  = Figure()
            plot = fig.add()

            #-----------------------------------------------------------------------------
            # Plot
            #-----------------------------------------------------------------------------

            plot_unit(i, sortedfile, plot)

            plot.xlabel('Time (ms)')
            plot.ylabel('Firing rate (a.u.)')

            props = {'prop': {'size': 8}, 'handletextpad': 1.02, 'labelspacing': 0.6}
            plot.legend(bbox_to_anchor=(0.18, 1), **props)

            #-----------------------------------------------------------------------------

            fig.save(path=p['figspath'],
                     name=p['name']+'_response_unit_{:03d}'.format(i))
            fig.close()

    #-------------------------------------------------------------------------------------
    # Selectivity
    #-------------------------------------------------------------------------------------

    elif action == 'selectivity':
        # Model
        m = p['model']

        trialsfile = get_trialsfile(p)
        dprime     = get_choice_selectivity(trialsfile)

        def get_first(x, p):
            return x[:int(p*len(x))]

        psig  = 0.25
        units = np.arange(len(dprime))
        try:
            idx = np.argsort(abs(dprime[m.EXC]))[::-1]
            exc = get_first(units[m.EXC][idx], psig)

            idx = np.argsort(abs(dprime[m.INH]))[::-1]
            inh = get_first(units[m.INH][idx], psig)

            idx = np.argsort(dprime[exc])[::-1]
            units_exc = list(exc[idx])

            idx = np.argsort(dprime[inh])[::-1]
            units_inh = list(units[inh][idx])

            units  = units_exc + units_inh
            dprime = dprime[units]
        except AttributeError:
            idx = np.argsort(abs(dprime))[::-1]
            all = get_first(units[idx], psig)

            idx    = np.argsort(dprime[all])[::-1]
            units  = list(units[all][idx])
            dprime = dprime[units]

        # Save d'
        filename = get_dprimefile(p)
        np.savetxt(filename, dprime)
        print("[ {}.do ] d\' saved to {}".format(THIS, filename))

        # Save selectivity
        filename = get_selectivityfile(p)
        np.savetxt(filename, units, fmt='%d')
        print("[ {}.do ] Choice selectivity saved to {}".format(THIS, filename))

    #-------------------------------------------------------------------------------------

    else:
        print("[ {}.do ] Unrecognized action.".format(THIS))
