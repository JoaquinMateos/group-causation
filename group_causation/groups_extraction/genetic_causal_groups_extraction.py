from collections import defaultdict
import random
import numpy as np
from typing import Any, Callable, cast

from deap import base, creator, tools, algorithms

from group_causation.groups_extraction.causal_groups_extraction import CausalGroupsExtractorBase
from group_causation.groups_extraction.stat_utils import get_scores_getter



class GeneticCausalGroupsExtractor(CausalGroupsExtractorBase): # Abstract class
    '''
    Class to extract a set of groups of variables by using an exhaustive search
    
    Args:
        data : np.array with the data, shape (n_samples, n_variables)
        scores : list of strings with the names of the scores to optimize
        scores_weights : list with the weights of the scores to optimize (a score of 1.0 means to maximize, -1.0 to minimize)
    '''
    def __init__(self, data: np.ndarray, scores: list[str], scores_weights: list, **kwargs):
        super().__init__(data, **kwargs)
        self.scores_getter = get_scores_getter(data, scores)
        self.scores_weights = scores_weights
    
    def extract_groups(self) -> list[set[int]]:
        '''
        Get score over all possible partitions of dataset and return the optimal one
        
        Returns
            groups : list of sets with the variables that compound each group
        '''
        best_partition = _run_genetic_algorithm(n_variables=self._data.shape[1],
                                                scores_getter=self.scores_getter,
                                                scores_weights=self.scores_weights)
        
        return best_partition    



def _run_genetic_algorithm(n_variables, scores_getter: Callable, scores_weights: list=[1.0]) -> list[set[int]]:
    # Define the set to partition
    ELEMENTS = list(range(0, n_variables))  # {1, 2, ..., N}

    # Genetic Algorithm: Define Fitness (Maximization)
    creator_mod = cast(Any, creator)
    if hasattr(creator_mod, "FitnessMax") and hasattr(creator_mod, "Individual"):
        del creator_mod.FitnessMax
        del creator_mod.Individual
    
    creator_mod.create("FitnessMax", base.Fitness, weights=scores_weights)
    creator_mod.create("Individual", list, fitness=creator_mod.FitnessMax)

    # Function to generate a random partition
    def random_partition():
        indices = list(range(n_variables))
        random.shuffle(indices)
        num_groups = random.randint(1, n_variables)  # Random number of subsets
        cuts = sorted(random.sample(range(1, n_variables), num_groups - 1))  # Cut points
        partition = []
        start = 0
        for cut in cuts + [n_variables]:
            partition.append({ELEMENTS[i] for i in indices[start:cut]})
            start = cut
        return partition

    # Register genetic operations
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator_mod.Individual, random_partition)
    toolbox_mod = cast(Any, toolbox)
    toolbox.register("population", tools.initRepeat, list, toolbox_mod.individual)

    toolbox.register("evaluate", scores_getter)

    toolbox.register("mate", crossover_partitions)

    # Custom mutation: move an element from one subset to another
    def mut_partition(individual):
        if len(individual) > 1:
            src_idx, dst_idx = random.sample(range(len(individual)), 2)
            if individual[src_idx]:  # Ensure source subset isn't empty
                element = individual[src_idx].pop()
                individual[dst_idx].add(element)
        return (individual,)

    toolbox.register("mutate", mut_partition)


    if len(scores_weights) == 1: # Single objective
        toolbox.register("select", tools.selTournament, tournsize=3)
    else: # Multi-objective; use a Pareto-based selection
        toolbox.register("select", tools.selNSGA2)
    
    # Run the Genetic Algorithm
    def run_ga():
        pop = toolbox_mod.population(n=min(100, max(5, 10 * n_variables)))
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=False)
        return pop
    
    # Execute GA and get best partition
    best_population = run_ga()
    best_individual = tools.selBest(best_population, k=1)[0]
    
    # Delete empty groups from the list
    best_individual = [group for group in best_individual if len(group) > 0]
    
    return best_individual


# Custom crossover: swap two random subsets
def crossover_partitions(part1, part2):
    '''
    Performs a crossover between two partitions, returning two new partitions.

    Parameters:
    - part1, part2: List of sets representing partitions of the same universal set.

    Returns:
    - Two new partitions as lists of sets (no overlaps, covering the entire universe).
    '''
    universe = set().union(*part1, *part2)
    elem_to_group1 = {}
    elem_to_group2 = {}

    # Randomly assign elements from part1 and part2 to offspring groupings
    for subset in part1:
        if random.random() < 0.5:
            for elem in subset:
                elem_to_group1[elem] = None  # Mark for now
    for subset in part2:
        if random.random() < 0.5:
            for elem in subset:
                elem_to_group2[elem] = None

    # Any missing elements go to the other mapping
    for elem in universe:
        if elem not in elem_to_group1:
            elem_to_group1[elem] = None
        if elem not in elem_to_group2:
            elem_to_group2[elem] = None

    # Reconstruct partitions by creating groupings
    groupings1 = defaultdict(set)
    groupings2 = defaultdict(set)

    # Assign elements into disjoint groups deterministically (e.g., by hash)
    for elem in universe:
        group_id1 = hash((elem, 'a')) % len(universe)
        group_id2 = hash((elem, 'b')) % len(universe)
        groupings1[group_id1].add(elem)
        groupings2[group_id2].add(elem)

    # Use only sets that contain assigned elements
    part1[:] = list(groupings1.values())
    part2[:] = list(groupings2.values())

    return part1, part2