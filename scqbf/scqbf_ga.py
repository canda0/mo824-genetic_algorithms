# scqbf/scqbf_Tabu Search.py

import math
from scqbf.scqbf_instance import *
from scqbf.scqbf_evaluator import *
import random
import time

class ScQbfGeneticAlgorithm:
    
    def __init__(self, instance: ScQbfInstance,
                 time_limit_secs: float = None,
                 generations: int = None,
                 pop_size: int = 5000,
                 mutation_rate: float = 0.5,
                 debug: bool = False,
                 config: dict = {
                    'stop_criteria': 'time' # Options: 'time', 'generations'
                 }):
        
        
        self.instance = instance
        self.config = config
        self.time_limit_secs = time_limit_secs
        self.debug = debug
        self.solve_time = 0
        self.generations = generations
        self.pop_size = pop_size
        self.chromosome_size = self.instance.n
        self.mutation_rate = mutation_rate
        self.current_generation = 0
        self.evaluator = ScQbfEvaluator(instance)

    def solve(self) -> ScQbfSolution:
        if self.instance is None:
            raise ValueError("Problem instance is not initialized")

        # Starts clock
        start_time = time.perf_counter()

        # Starts Initial Population
        population = self.initialize_population()
        best_chromosome = self.get_best_chromosome(population)
        best_sol = self.decode(best_chromosome)

        if self.debug:
            print(f"Generation {self.current_generation}: BestSolution =", best_sol)

        # Loop for each generation
        # for g in range(1, self.generations + 1):
        while ((self.current_generation < self.generations) if self.config.get('stop_criteria', 'time') == 'generations' else True):
            # Update generation
            self.current_generation += 1

            # Select parents
            parents = self.select_parents(population)

            # Generate offsprings throught crossover
            offsprings = self.crossover(parents)

            # Apply mutation
            mutants = self.mutate(offsprings)

            # Select best individuals for the next generation
            new_population = self.select_population(mutants, best_chromosome)
            population = new_population

            # Get best solution in the current population
            best_chromosome = self.get_best_chromosome(population)

            # Update best solution found so far
            best_fitness = self.fitness(best_chromosome)
            if best_fitness._last_objfun_val > best_sol._last_objfun_val:
                best_sol = self.decode(best_chromosome)
                if self.debug:
                    print(f"Generation {self.current_generation}: BestSolution =", best_sol)
            
            # Check time limit
            self.solve_time = time.perf_counter() - start_time
            if self.config.get('stop_criteria', 'time') == 'time' and self.solve_time >= self.time_limit_secs:
                print(f"Time limit of {self.time_limit_secs} seconds reached, stopping Genetic Algorithm in generation {self.current_generation}.")
                break

        return best_sol

    def initialize_population(self) -> List[List[int]]:
        """
        Initializes a population of random chromosomes.
        Each chromosome is represented as a list in {0, 1} where each gene indicates the inclusion (1) or exclusion (0) 
        of an element in the solution. The population consists of 'self.pop_size' chromosomes, each of length 'self.chromosome_size'.
        ----------
        Returns:
            List[List[int]]: A list containing the randomly generated chromosomes.
        """
        population = []
        while len(population) < self.pop_size:
            chromosome = [random.randint(0, 1) for _ in range(self.chromosome_size)]
            population.append(chromosome)
        return population

    def decode(self, chromosome: List[int]) -> ScQbfSolution:
        """
        Decodes a chromosome into a ScQbfSolution.
        ----------
        Parameters
            chromosome (List[int]): A list representing the chromosome to be decoded.
        Returns:
            float: The decoded solution containing the selected elements.
        """
        sol_elements = [i for i in range(len(chromosome)) if chromosome[i] == 1]
        sol = ScQbfSolution(sol_elements)
        sol._last_objfun_val = self.evaluator.evaluate_objfun(sol)
        return sol
        # return ScQbfSolution(sol_elements)

    def fitness(self, chromosome: List[int]) -> float:
        """
        Evaluates the fitness of a given chromosome.
        ----------
        Parameters
            chromosome (List[int]): A list representing the genes of the chromosome to be evaluated.
        Returns:
            float: The fitness value of the chromosome.
        """
        sol = self.decode(chromosome)
        return sol
        # return self.evaluator.evaluate_objfun(sol)

    def get_best_chromosome(self, population: List[List[int]]) -> List[int]:
        """
        Identifies and returns the chromosome with the highest fitness value from the population.
        ----------
        Parameters:
            population (List[List[int]]): A list of chromosomes, where each chromosome is a list of genes (0s and 1s).
        Returns:
            best_chromosome (List[int]): The chromosome with the highest fitness value.
        """
        best_chromosome = None
        best_fitness = float("-inf")

        for chromosome in population:
            fitness = self.fitness(chromosome)
            if fitness._last_objfun_val > best_fitness:
                best_fitness = fitness._last_objfun_val
                best_chromosome = chromosome

        return best_chromosome
    
    def select_parents(self, population: List[List[int]]) -> List[List[int]]:
        """
        Selection of parents for crossover using the tournament method. 
        Given a population of chromosomes, randomly takes two chromosomes and compare them by their fitness. 
        The best one is selected as parent. Repeat until the number of selected parents is equal to 'self.pop_size'.
        ----------
        Parameters:
            population (List[List[int]]): A list of chromosomes
        Returns:
            parents (List[List[int]]): A list of selected parent chromosomes.
        """
        parents = []

        while len(parents) < self.pop_size:
            # Sort two random chromosomes as parents
            index_p1 = random.randint(0, self.pop_size - 1)
            index_p2 = random.randint(0, self.pop_size - 1)

            if index_p1 != index_p2:
                parent_1 = population[index_p1]
                parent_2 = population[index_p2]

                # Select parent with best fitness
                if self.fitness(parent_1)._last_objfun_val > self.fitness(parent_2)._last_objfun_val:
                    parents.append(parent_1)
                else:
                    parents.append(parent_2)

        return parents
    
    def crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """
        The crossover step takes the parents generated by 'self.select_parents' and recombine their genes to generate new chromosomes (offsprings). 
        The method being used is the 2-point crossover, which randomly selects two locus for being the points of exchange (P1 and P2). 
        ----------
        Parameters:
            parents (List[List[int]]): A list of parent chromosomes.
        Returns:
            offsprings (List[List[int]]): A list of offspring chromosomes generated from the parents.
        """
        offsprings = []

        # Apply crossover for each pair of parents
        for i in range(0, len(parents), 2): 
            parent_1 = parents[i]
            parent_2 = parents[i + 1]

            # Select random crosspoints
            crosspoint1 = random.randint(0, self.chromosome_size)
            crosspoint2 = crosspoint1 + random.randint(0, (self.chromosome_size + 1) - crosspoint1)

            # Create offsprings by exchanging genes between parents
            offspring_1 = parent_1[:crosspoint1] + parent_2[crosspoint1:crosspoint2] + parent_1[crosspoint2:]
            offspring_2 = parent_2[:crosspoint1] + parent_1[crosspoint1:crosspoint2] + parent_2[crosspoint2:]

            # Add offsprings to the new population
            offsprings.append(offspring_1)
            offsprings.append(offspring_2)
        
        return offsprings
    
    def mutate(self, offsprings: List[List[int]]) -> List[List[int]]:
        """
        The mutation step introduces random changes to the genes of the offsprings generated by 'self.crossover'. 
        Each gene in each offspring has a probability of 'self.mutation_rate' to be flipped (0 to 1 or 1 to 0).
        ----------
        Parameters:
            offsprings (List[List[int]]): A list of offspring chromosomes.
        Returns:
            mutated_offsprings (List[List[int]]): A list of mutated offspring chromosomes.
        """
        mutated_offsprings = []

        for chromosome in offsprings:
            mutated_offspring = [
                gene if random.random() > self.mutation_rate else 1 - gene
                for gene in chromosome
            ]
            mutated_offsprings.append(mutated_offspring)

        return mutated_offsprings
    
    def get_worst_chromosome(self, population: List[List[int]]) -> List[int]:
        """
        Identifies and returns the chromosome with the lowest fitness value from the population.
        ----------
        Parameters:
            population (List[List[int]]): A list of chromosomes, where each chromosome is a list of genes (0s and 1s).
        Returns:
            worst_chromosome (List[int]): The chromosome with the lowest fitness value.
        """
        worst_chromosome = None
        worst_fitness = float("inf")

        for chromosome in population:
            fitness = self.fitness(chromosome)
            if fitness._last_objfun_val < worst_fitness:
                worst_fitness = fitness._last_objfun_val
                worst_chromosome = chromosome

        return worst_chromosome

    def select_population(self, offsprings: List[List[int]], best_chromosome: List[int]) -> List[List[int]]:
        """
        Updates the population that will be considered for the next generation. 
        The method used for updating the population is the elitist,
	    which simply takes the worse chromosome from the offsprings and replace
	    it with the best chromosome from the previous generation.
        ----------
        Parameters:
            offsprings (List[List[int]]): A list of mutated offspring chromosomes.
            best_chromosome (List[int]): The best chromosome from the previous generation.
        Returns:
            new_population (List[List[int]]): A list of chromosomes selected for the next generation.
        """
        worse = self.get_worst_chromosome(offsprings)

        if self.fitness(worse)._last_objfun_val < self.fitness(best_chromosome)._last_objfun_val:
            offsprings.remove(worse)
            offsprings.append(best_chromosome)

        return offsprings