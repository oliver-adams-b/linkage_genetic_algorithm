"""
Creating the genetic algorith to create the tower that draws a given curve!
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import linkage
from progress_bar import ProgressBar
from scipy.spatial.distance import directed_hausdorff
import random

"""
Define the fitness function!
"""
def l2_fitness(tower_trace, target_trace):
    partial_sum = 0
    
    target_trace = target_trace[:10:-1]
    tower_trace = tower_trace[:10:-1]
    
    target_trace = target_trace[:500]
    tower_trace = tower_trace[:500]
    
    if (len(tower_trace) <= 1) or (len(target_trace) <= 1):
        return 10000
    
    for i in range(len(tower_trace)):
        tow = np.asarray(tower_trace[i])
        tar = np.asarray(target_trace[i])

        partial_sum = partial_sum + np.linalg.norm(tow - tar)**2
        
    return partial_sum / len(tower_trace)


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
    
    if len(test) <= 300:
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
    endpoint_fitness = np.mean([start_point_dist, end_point_dist])/4

    return  hausdorff_fitness + endpoint_fitness 
    
    
def reproduce(parent_tower, 
              num_children, 
              mutation_factor, 
              num_randos, 
              randos_mutation_factor):
    
    """
    Takes a parent, and creates the next generation stemming from that parent. 
    We introduce some randos in the mix just to make sure that the population
    always has some randos in it! 
    """
    
    children_L = []
    parent_n_quads = len(parent_tower)
    
    for i in range(num_children):
        temp_L = []
        
        for i in range(parent_n_quads):
            parent_L = np.abs(list(np.asarray(parent_tower[i]) + (np.random.rand(4)*2 -1)*mutation_factor))
            temp_L.append(parent_L)
        
        children_L.append(temp_L)
        
    randos_L = generate_randos(num_randos,  num_quads = parent_n_quads, 
                               randos_mutation_factor = randos_mutation_factor)
    
    for i in range(num_randos):
        children_L.append(randos_L[i])
        
    children_L.append(parent_tower)
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

def get_best_of_generation(A0, A1, tower_gen, target_trace):
    """
    tower_gen is a list of towers, and target_trace is the trace we are hoping
    the towers in tower_gen begin to draw. We compute the trace for each tower
    in tower_gen, and spit out the one with the lowest fitness factor. 
    """
    best_fit = 100
    best_tower = None
    
    progress = ProgressBar(len(tower_gen), fmt = ProgressBar.FULL)
    
    for tower in tower_gen:
        progress.current += 1
        progress.__call__()
        
        temp_trace = linkage.get_trace(A0, A1, tower, 1000)
        #fitness = l2_fitness(temp_trace, target_trace)
        fitness = hausdorff_like_fitness(temp_trace, target_trace)
        
        if fitness < best_fit:
            best_fit = fitness
            best_tower = tower
    
    print("\n")
    #progress.done()
    
    return best_fit, best_tower

def get_best_n_of_generation(A0, A1, tower_gen, target_trace, n):
    """
    the same as get_best_of_generation(), but now it returns the n towers with the
    lowest fitness. We take advantage of pandas nlowest() function to help out. 
    """
    lookup_df = pd.DataFrame(columns = ["tower", "fitness"], index = [0])
    
    progress = ProgressBar(len(tower_gen), fmt = ProgressBar.FULL)
    
    for index, tower in enumerate(tower_gen):
        progress.current += 1
        progress.__call__()
        
        temp_trace = linkage.get_trace(A0, A1, tower, 1000)
        fitness = hausdorff_like_fitness(temp_trace, target_trace)
        
        temp_df = pd.DataFrame({"tower":[tower], "fitness":fitness}, index = [0])
        lookup_df = lookup_df.append(temp_df, sort = False, 
                                     ignore_index = False)
    
    #best_fitness = np.min(lookup_df['fitness'])
    lookup_df = lookup_df.astype({"fitness":"float32"})
    lookup_df = lookup_df.nsmallest(n, "fitness")

    return lookup_df['fitness'].values, lookup_df['tower'].values


def run_sim(A0, A1, 
            target_trace, 
            n_best, 
            n_quads, 
            n_generations, 
            n_children_per_gen, 
            init_pool_size):
    """
    Initialize first generation:
    """
    first_gen = generate_randos(init_pool_size, 
                                num_quads = n_quads, 
                                randos_mutation_factor = 0.3)
    
    
    print("Computing traces and fitness for the first generation:")
    best_fit, best_towers = get_best_n_of_generation(p0, p1, 
                                                     first_gen, 
                                                     target_trace, n_best*3)
    next_generation_L = best_towers
    print("\nBest Fitness: {}".format(best_fit))
    
    """
    From the best performing tower in the previous generation, compute the next gen
    and update the previous best. Save past bests!
    """
    best_fits = []
    
    for n in range(n_generations):
        
        print("\nComputing traces and fitness of generation {}/{}:".format(n+1, n_generations))
        
        temp_best_fit, temp_best_towers = get_best_n_of_generation(p0, p1, 
                                                                   next_generation_L, 
                                                                   target_trace, n_best)
        next_generation_L = []
        
        for tower in temp_best_towers:
            temp_children_L = reproduce(tower, 
                                        num_children = n_children_per_gen, 
                                        mutation_factor = 0.08 * (0.9999**(n)), 
                                        num_randos = 2, 
                                        randos_mutation_factor = 0.3)
            
            for child_L in temp_children_L:
                next_generation_L.append(child_L)
                                                          
        best_tower = temp_best_towers
        best_fits.append(temp_best_fit)
        print("\n{}".format(temp_best_fit))
        plt.plot(best_fits)
        plt.xlabel("generation number")
        plt.ylabel("fitness")
        plt.show()
        
        if n%5 == 0:
            for tower in best_tower:
                linkage.disp_trace(linkage.get_trace(p0, p1, tower, 1000), 
                                   cmap = 'autumn', label = 'best_trace')
                linkage.disp_trace(target_trace, label = 'target_trace')
                plt.show()
            
    linkage.disp_tower(best_tower[0])
    linkage.disp_trace(linkage.get_trace(p0, p1, best_tower[0], 1000), 
                       cmap = 'autumn', label = 'best_trace')
    linkage.disp_trace(target_trace, label = 'target_trace')
    
    plt.show()
    
    return best_tower[0]


"""
Initialize the anchor points and target structures
"""
p0 = np.asarray([0, 0])
p1 = np.asarray([1, 0])
n_best = 5
n_quads = 6
n_generations = 150
n_children_per_gen = 15
init_pool_size = 200

test_target_L = np.loadtxt("target_l.csv")
test_target_trace = np.loadtxt("target_trace.csv")

"""
Trying to get the linkages to draw a heart, 
splitting up the two lobes. 
"""


"""
The ability of the linkage to approximate the curve greately depends on the spatial
position of the curve that it is trying to approximate. 
"""
left_heart = linkage.make_heart(1000)[:500] + [1, 2.5]
linkage.disp_trace(left_heart)


best_lh_tower = run_sim(A0 = p0, A1 = p1, 
                     target_trace = left_heart, 
                     n_best = n_best, 
                     n_quads = n_quads, 
                     n_generations = n_generations, 
                     n_children_per_gen = n_children_per_gen, 
                     init_pool_size = init_pool_size)

np.savetxt("best_lh_tower.csv", best_lh_tower)


right_heart = (linkage.make_heart(1000)[500:] + [.2, 8])[-1:-500:-1]
linkage.disp_trace(left_heart)

best_rh_tower = run_sim(A0 = p0, A1 = p1, 
                        target_trace = right_heart, 
                        n_best = n_best, 
                        n_quads = n_quads, 
                        n_generations = n_generations, 
                        n_children_per_gen = n_children_per_gen, 
                        init_pool_size = init_pool_size))

np.savetxt("best_rh_tower.csv", best_rh_tower)



