"""
Creating the genetic algorithm to create the tower that draws a provided curve!
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from graphs import Graph
import os

#personal libraries
import linkage
from progress_bar import ProgressBar



def hausdorff_like_fitness(test, target_temp):
    """
    Uses the directed hausdorff distance of the test trace to the 
    target trace. Convergence is aided by also measuring the 
    distance between the hausdorff distance between the derivatives
    of the traces. Finally, the fitness is futher penalized by the
    distance between the respective start and endpoints of the target
    and test traces. 
    """
    target = target_temp
    
    if (type(test) == type(None)) or (len(test) <= 1):
        return 10000
    
    if len(test) <= len(target)/3:
        """
        if the test_trace does not have enough data, then toss it
        """
        return 10000
    
    """
    We want to make the traces the same size, without cutting off chunks of 
    the traces. So we take an evenly-spaced subsample of the one that is larger, 
    so that we maintain the overall shape, just with fewer points. 
    """
        
    if len(test) >= len(target):
        step = int(np.ceil(len(test)/len(target)))
        test = test[:-1:step]
        test = test[:len(target)]
        
        if len(test) <= 1:
            return 10000
        
    if len(target) >= len(test):
        step = int(np.ceil(len(target)/len(test)))
        target = target[:-1:step]
        target = target[:len(test)]
        
        if len(target) <= 1:
            return 10000
        
    start_point_dist = np.linalg.norm(test[0] - target[0])
    end_point_dist = np.linalg.norm(test[-1] - target[-1])

    
    p =  directed_hausdorff(test, 
                            target)[0]
    q =  directed_hausdorff(target, 
                            test)[0]
    
    p_prime = directed_hausdorff(np.gradient(test)[0], 
                                 np.gradient(target)[0])[0]
    q_prime = directed_hausdorff(np.gradient(target)[0], 
                                 np.gradient(test)[0])[0]
    

    hausdorff_fitness = max(p, q) + max(p_prime, q_prime)
    endpoint_fitness = np.mean([start_point_dist, end_point_dist])/2
    
    """
    The units for hausdorff_fitness and endpoint_fitness are both distance.
    We care equally about the 'distance' between the traces themselves as much
    as we care about the 'distance' between the derivatives of the traces. 
    
    The endpoint_fitness is well-defined, and ensures that the generated traces
    are penalized for not having the trace start and end at the correct locations.
    """
    return  hausdorff_fitness + endpoint_fitness 
    
    
def make_offspring(parent_tower, 
                   num_children, 
                   mutation_factor, 
                   num_randos, 
                   randos_mutation_factor, 
                   n_adjusted_children = 0):
    
    """
    Takes a parent, and creates the next generation stemming from that parent. 
    We introduce some randos in the mix just to make sure that the population
    always has some randos in it! 
    
    n_adjusted_children is the number of childen to create that are scalings and
    pencil_adjustments of the parent_tower. So we are either scaling the whole 
    parent tower or just the last two lengths. 
    """
    
    children_L = []
    parent_n_quads = len(parent_tower)
    
    for i in range(num_children):
        temp_L = []
        
        for i in range(parent_n_quads):
            parent_L = list(np.abs(np.asarray(parent_tower[i]) + (np.random.rand(4)*2 -1)*mutation_factor))
            temp_L.append(parent_L)
        
        children_L.append(temp_L)
        
    randos_L = generate_randos(num_randos,  num_quads = parent_n_quads, 
                               randos_mutation_factor = randos_mutation_factor)
    
    for i in range(num_randos):
        children_L.append(randos_L[i])
        
    children_L.append(parent_tower)
    
    n_pencil_adjusts = int(n_adjusted_children/3)
    n_scalar_adjusts = n_adjusted_children - n_pencil_adjusts
    
    for _ in range(n_pencil_adjusts):
        children_L.append(linkage.adjust_pencil(parent_tower, 
                                                (np.random.rand(1)*2)[0]))
    for _ in range(n_scalar_adjusts):
        children_L.append(linkage.scale_tower(parent_tower, 
                                              (np.random.rand(1)*1.5 + 0.5)[0]))
    return children_L
  
def crossover(parents):
    """
    For each pair of parents, return the average of the two! Putting a little bit of cross-over
    in the population. If there are n parents, there will be n(n-1)/2 offspring
    """
    
    def make_edges(n):
        """
        Returns the enumerated edges of a complete graph with n vertices, a list
        of the form [[i, j], ...]
        """
        edges  = []
        for i in range(n):
            for j in range(i, n):
                edges.append([i, j])
        return edges
    
    def get_mean_of_parents(parent1, parent2):
        return np.mean([parent1, parent2], axis = 0).tolist()
    
    edges = make_edges(len(parents))
    children_L = []
    
    for edge in edges:
        children_L.append(get_mean_of_parents(parents[edge[0]], 
                                              parents[edge[1]]))
    return children_L
        
        
def generate_randos(num_randos, 
                    num_quads, 
                    randos_mutation_factor = 0.01):
    """
    Generates a set of random towers with a given mutation factor
    """
    randos_L = []

    for n in range(num_randos):
        L = []
        
        for i in range(num_quads):
            L.append(list((np.random.rand(4)*2 -1)*randos_mutation_factor + 1))
            
        randos_L.append(L)
        
    return randos_L

def get_best_of_generation(A0, 
                           tower_gen, 
                           target_trace, 
                           n_sample_points = 1000):
    """
    tower_gen is a list of towers, and target_trace is the trace we are hoping
    the towers in tower_gen begin to draw. We compute the trace for each tower
    in tower_gen, and spit out the one with the lowest fitness factor. 
    """
    best_fit = 10000
    best_tower = None
    
    progress = ProgressBar(len(tower_gen), fmt = ProgressBar.FULL)
    
    for tower in tower_gen:
        progress.current += 1
        progress.__call__()
        
        temp_trace = linkage.get_trace(A0, tower, n_sample_points)

        fitness = hausdorff_like_fitness(temp_trace, target_trace)
        
        if fitness < best_fit:
            best_fit = fitness
            best_tower = tower
    
    print("\n")
    
    return best_fit, best_tower

def get_best_n_of_generation(A0, 
                             tower_gen, 
                             target_trace, 
                             n, 
                             n_sample_points = 1000):
    """
    the same as get_best_of_generation(), but now it returns the n towers with the
    lowest fitness. We take advantage of pandas nlowest() function to help out. 
    """
    lookup_df = pd.DataFrame(columns = ["tower", "fitness"], index = [0])
    
    progress = ProgressBar(len(tower_gen), fmt = ProgressBar.FULL)
    
    for index, tower in enumerate(tower_gen):
        progress.current += 1
        progress.__call__()
        
        temp_trace = linkage.get_trace(A0, tower, n_sample_points)
        fitness = hausdorff_like_fitness(temp_trace, target_trace)
        
        temp_df = pd.DataFrame({"tower":[tower], "fitness":fitness}, index = [0])
        lookup_df = lookup_df.append(temp_df, sort = False, 
                                     ignore_index = False)
    
    lookup_df = lookup_df.astype({"fitness":"float32"})
    lookup_df = lookup_df.nsmallest(n, "fitness")

    return lookup_df['fitness'].values, lookup_df['tower'].values


def run_sim(A0, 
            target_trace, 
            n_best, 
            n_quads, 
            n_generations, 
            n_children_per_gen, 
            init_pool_size, 
            n_sample_points = 800, 
            show_traces = False):
    """
    Initialize first generation:
    """
    first_gen = generate_randos(init_pool_size, 
                                num_quads = n_quads, 
                                randos_mutation_factor = 0.3)
    
    
    print("Computing traces and fitness for the first generation:")
    best_fit, best_towers = get_best_n_of_generation(A0, 
                                                     first_gen, 
                                                     target_trace, n_best**2, 
                                                     n_sample_points = n_sample_points)
    next_generation_L = best_towers
    print("\nBest Fitness: {}".format(best_fit))
    
    """
    From the best performing towers in the previous generation, compute the next gen
    and update the previous best. 
    """
    best_fits = []
    
    for n in range(n_generations):
        
        print("\nComputing traces and fitness of generation {}/{}:".format(n+1, n_generations))
        
        temp_best_fit, temp_best_towers = get_best_n_of_generation(A0, 
                                                                   next_generation_L, 
                                                                   target_trace, n_best)
        next_generation_L = []
        
        for tower in temp_best_towers:
            temp_children_L = make_offspring(tower, 
                                             num_children = n_children_per_gen, 
                                             mutation_factor = 0.08, 
                                             num_randos = 2, 
                                             randos_mutation_factor = 0.3)
            
            for child_L in temp_children_L:
                next_generation_L.append(child_L)
                 
        """
        Plot how well the computer is doing at each time step.
        """                                        
        best_tower = temp_best_towers
        best_fits.append(temp_best_fit)
        print("\n{}".format(temp_best_fit))
        plt.plot(best_fits)
        plt.ylim(bottom = 0)
        plt.hlines(0.25,
                   xmin = 0, 
                   xmax = n+1,
                   linestyles = 'dotted', 
                   colors = 'r')
        plt.xlabel("Generation Number")
        plt.ylabel("Hausdorff-like Distance")
        plt.title("Distance from the Target Trace For \n the Best {} Agents over {} Generations".format(n_best, n_generations))
        plt.show()
        
        if (n%5 == 0) and show_traces:
            '''
            Every five steps, draw the n_best traces to get a visual about 
            how the decreases in fitness match with the best-fit traces. 
            '''
            for tower in best_tower:
                linkage.disp_trace(linkage.get_trace(A0, tower, int(n_sample_points/2)), 
                                   cmap = 'autumn', label = 'best_trace')
                linkage.disp_trace(target_trace, label = 'target_trace')
                plt.show()
                
        # if n > 150:
        #     #if no progress has been made in the past 60 generations, break
        #     if best_fits[n][0] == best_fits[n - 100][0]:
        #         break
            
       
    '''
    At the end, show the monstrosity of a tower that the computer has made for you
    '''
    linkage.disp_tower(best_tower[0])
    linkage.disp_trace(linkage.get_trace(A0, best_tower[0], n_sample_points), 
                       cmap = 'autumn', label = 'best_trace')
    linkage.disp_trace(target_trace, label = 'target_trace')
    
    plt.show()
    
    return best_tower[0]


def gen_towers_from_traces(traces, 
                           save_folder_name):
    """
    Given a set of traces, generate some towers whose traces approximate those
    that are given. Save the best towers for each trace
    """
    p0 = np.asarray([0, 0])
    n_best = 5
    n_quads = 6
    n_generations = 450
    n_children_per_gen = 15
    init_pool_size = 300
    
    try:
        os.mkdir(save_folder_name)
    except FileExistsError:
        pass
    
    for index, trace in enumerate(traces):
        print("\nWorking on trace {} out of {} ********************* \n".format(index, len(traces)))
        temp_tower = run_sim(A0 = p0, 
                             target_trace = trace, 
                             n_best = n_best, 
                             n_quads = n_quads, 
                             n_generations = n_generations, 
                             n_children_per_gen = n_children_per_gen, 
                             init_pool_size = init_pool_size)
        
        np.savetxt("{}/trace_tower_{}.csv".format(save_folder_name, index), temp_tower)
    
    

dna = linkage.make_dna(800, dpos = [1, 5.5], dscale = [1.15, 0.7])
gen_towers_from_traces(dna, 'dna_output')
    
"""
'''
Trying to get the linkages to draw a heart, 
splitting up the two lobes. 

The ability of the linkage to approximate the curve greately 
depends on the spatial position of the curve that it is trying to approximate. 
'''

'''
Initialize the anchor points and 'hyperparameters'
'''
p0 = np.asarray([0, 0])
n_best = 5
n_quads = 5
n_generations = 250
n_children_per_gen = 15
init_pool_size = 200

left_heart = (linkage.make_heart(1600)[:800] + [2, 10])
linkage.disp_trace(left_heart)

best_lh_tower = run_sim(A0 = p0, 
                      target_trace = left_heart, 
                      n_best = n_best, 
                      n_quads = n_quads, 
                      n_generations = n_generations, 
                      n_children_per_gen = n_children_per_gen, 
                      init_pool_size = init_pool_size)

np.savetxt("best_lh_tower.csv", best_lh_tower)


right_heart = (linkage.make_heart(1600)[800:] + [.5, 10])[-1:0:-1]
linkage.disp_trace(right_heart)

best_rh_tower = run_sim(A0 = p0, 
                      target_trace = right_heart, 
                      n_best = n_best, 
                      n_quads = n_quads, 
                      n_generations = n_generations, 
                      n_children_per_gen = n_children_per_gen, 
                      init_pool_size = init_pool_size)

np.savetxt("best_rh_tower.csv", best_rh_tower)
"""

