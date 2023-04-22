#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Global Optimization and Intrusion Detection Using a Novel Gaussian Mutated and Shrink Moth Flame Optimization
# More details about the algorithm are in [please cite the original paper ] DOI : 
# Nazar K. Hussein , Mohammed Qaraad   , Souad Amjad  , M.A.Farag  , Saima Hassan  , Seyedali Mirjalili  and Mostafa A. Elhosseini , "Global Optimization and Intrusion Detection Using a Novel Gaussian Mutated and Shrink Moth Flame Optimization"
# Journal of Computational Design and Engineering, 2023# -*- coding: utf-8 -*-
"""
@author: Mohammed Qaraad
"""

import random
import random as rande
import numpy as np
import numpy
import math
import matplotlib.pyplot as plt

from numpy.random import uniform, normal, randint, rand
from numpy import exp, pi, sin, cos, zeros, minimum, maximum, abs, where, sign, mean, stack
from numpy import min as np_min
from numpy import max as np_max
from copy import deepcopy

from numpy import where, clip, logical_and, maximum, minimum, power, sin, abs, pi, sqrt, sign, ones, ptp, min, sum, array, ceil, multiply, mean
from numpy.random import uniform, random, normal, choice
from math import gamma



ID_MIN_PROB = 0  # min problem
ID_MAX_PROB = -1  # max problem
ID_POS = 0  # Position
ID_FIT = 1  # Fitness

EPSILON = 10E-10
DEFAULT_BATCH_IDEA = False
DEFAULT_BATCH_SIZE = 10
DEFAULT_LB = -1
DEFAULT_UB = 1
    # Alias for C function


def GMSMFO(objf, lb, ub, dim, SearchAgents_no, Max_iter,verbose=False):

    Convergence_curve = numpy.zeros(Max_iter)
 
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim    
    epochs = Max_iter
    pop_size = SearchAgents_no

    epochs = Max_iter
    problem_size = dim
    ID_WEI = 2
 

    pop_moths = [create_solution(lb,ub,objf,problem_size) for _ in range(SearchAgents_no)]
   
    pop_flames, g_best = get_sorted_pop_and_global_best_solution(pop=pop_moths, id_fit=ID_FIT, id_best=ID_MIN_PROB)     # Eq.(2.6)
    problem_size = dim
    
    for epoch in range(epochs):
            beta = 0.2+(1.2-0.2)*(1-(epoch/epochs)**3)**2    #                       % Eq.(14.2)
            alpha = abs(beta*math.sin((3*math.pi/2+math.sin(3*math.pi/2*beta))));  #  

            # Number of flames Eq.(3.14) in the paper (linearly decreased)
            num_flame = round(SearchAgents_no - (epoch + 1) * ((SearchAgents_no - 1) / epochs))

            # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a = -1 + (epoch + 1) * ((-1) / epochs)
            shrink=2*(1-epoch/epochs)
            for i in range(SearchAgents_no):
                r = rande.random()
                #   D in Eq.(3.13)
                distance_to_flame = abs(pop_flames[i][ID_POS] - pop_moths[i][ID_POS])
                t = (a - 1) * uniform(0, 1, problem_size) + 1
                b = 1
                vb = 2*shrink*r-shrink
                # Update the position of the moth with respect to its corresponding flame, Eq.(3.12).
                temp_1 = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + +vb * pop_flames[i][ID_POS]

                # Update the position of the moth with respect to one flame Eq.(3.12).
                ## Here is a changed, I used the best position of flames not the position num_flame th (as original code)
                temp_2 = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + g_best[ID_POS]

                list_idx = i * ones(problem_size)
                pos_new = where(list_idx < num_flame, temp_1, temp_2)
                
                Xnew = numpy.zeros((1,problem_size))

                M = sum(pos_new)/dim;
                Sig=(sum((pos_new-M)**2)/dim)**0.5
                if Sig==0:
                    Sig=0.0000000000001
                Y = numpy.full(dim,float(0))
                Y = (1/numpy.sqrt(2*math.pi*Sig**2))*numpy.exp(-(pos_new-M)**2/2*Sig**2); 
            
                Y = where(uniform(0, 1, dim) < 0.5, pos_new, Y)
                Y = numpy.clip(Y, lb, ub)
                if objf(Y)<pop_moths[i][ID_FIT]:
                    pos_new=Y;                  
                
                pos_new = amend_position_faster(pos_new,lb,ub)
            
                              
                ## This is the way I make this algorithm working. I tried to run matlab code with large dimension and it will not convergence.
                fit_new = get_fitness_position(objf,pos_new)
                if fit_new < pop_moths[i][ID_FIT]:
                    pop_moths[i] = [pos_new, fit_new]

            # Update the global best flame
            pop_flames = pop_flames + pop_moths
            pop_flames, g_best = update_sorted_population_and_global_best_solution(pop_flames, ID_MIN_PROB, g_best)
            pop_flames = pop_flames[:SearchAgents_no]

            Convergence_curve[epoch] = g_best[ID_FIT]
            if verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[ID_FIT]))
    return Convergence_curve

def update_sorted_population_and_global_best_solution(pop=None, id_best=None, g_best=None):
    """ Sort the population and update the current best position. Return the sorted population and the new current best position """
    sorted_pop = sorted(pop, key=lambda temp: temp[ID_FIT])
    current_best = sorted_pop[id_best]
    g_best = deepcopy(current_best) if current_best[ID_FIT] < g_best[ID_FIT] else deepcopy(g_best)
    return sorted_pop, g_best
def amend_position_faster(position, lb, ub):
    return clip(position, lb, ub)

def create_solution(lb , ub, obj_func,dim,minmax=0):
    """ Return the position position with 2 element: position of position and fitness of position

        Parameters
        ----------
        minmax
            0 - minimum problem, else - maximum problem
    """
    position = uniform(lb,ub)
    fitness = get_fitness_position(obj_func, position=position, minmax=minmax)
    weight = zeros(dim)
    return [position, fitness,weight]

def get_sorted_pop_and_global_best_solution(pop=None, id_fit=None, id_best=None):
    """ Sort population and return the sorted population and the best position """
    sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
    return sorted_pop, deepcopy(sorted_pop[id_best])


def get_fitness_position(obj_func,position, minmax=0):
            
    """     Assumption that objective function always return the original value
        :param position: 1-D numpy array
        :param minmax: 0- min problem, 1 - max problem
        :return:
        """
    fit_new = obj_func(position)
    return fit_new if minmax == 0 else 1.0 / (fit_new + EPSILON)
def weighted_lehmer_mean(list_objects, list_weights):
    up = list_weights * list_objects ** 2
    down = list_weights * list_objects
    return sum(up) / sum(down)

def update_global_best_solution(pop=None, id_best=None, g_best=None):
    """ Sort the copy of population and update the current best position. Return the new current best position """
    sorted_pop = sorted(pop, key=lambda temp: temp[ID_FIT])
    current_best = sorted_pop[id_best]
    return deepcopy(current_best) if current_best[ID_FIT] < g_best[ID_FIT] else deepcopy(g_best)
def get_global_best_solution(pop=None, id_fit=None, id_best=None):
    """ Sort a copy of population and return the copy of the best position """
    sorted_pop = sorted(pop, key=lambda temp: temp[id_fit])
    return deepcopy(sorted_pop[id_best])

def objective_Fun(x):
    dim = len(x)
    o = numpy.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o

######  main 
Max_iterations=50  # Maximum Number of Iterations
swarm_size = 30 # Number of salps
LB=-30  #lower bound of solution
UB=30   #upper bound of solution
Dim=50 #problem dimensions
NoRuns=10  # Number of runs
ConvergenceCurve=np.zeros((Max_iterations,NoRuns))
for r in range(NoRuns):
    result = GMSMFO(objective_Fun, LB, UB, Dim, swarm_size, Max_iterations)
    ConvergenceCurve[:,r]=result
# Plot the convergence curves of all runs
idx=range(Max_iterations)
fig= plt.figure()

#3-plot
ax=fig.add_subplot(111)
for i in range(NoRuns):
    ax.plot(idx,ConvergenceCurve[:,i])
plt.title('Convergence Curve of the GMSMFO Optimizer', fontsize=12)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.show()

