#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 19:53:41 2018

for shelving all variables in the current workspace from globals()

@author: aiden
"""
import shelve
import os


#=========================================================================================
# Command line
#=========================================================================================
class shelb(object):
    
    def __init__(self, shle):
        
        self.file = shle
        if not self.file.endswith('.py'):
            self.file += '.py'
   
    def save(self):
        
        # shelve file
        shelf_file = os.path.abspath(self.file)
        save_shelf = shelve.open(shelf_file,'n')    # 'n' is for saving or creating new shelf
        
        for key in dir():
            try:
                save_shelf[key] = globals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR <TYPE>     shelving: {0}'.format(key))
            except:
                print('ERROR <????>     shelving: {0}'.format(key))
        save_shelf.close()
                        
    def load(self):
    
        shelf_file = os.path.abspath(self.file)
        load_shelf = shelve.open(shelf_file)          # open existing shelf
        
        for key in load_shelf:
            try:
                globals()[key] = load_shelf[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR <TYPE>     shelving: {0}'.format(key))
            except:
                print('ERROR <????>     shelving: {0}'.format(key))
        load_shelf.close()