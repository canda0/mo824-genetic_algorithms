# Versão que cria um processo por GA. Talvez paralelizar run_experiment para usar mais núcleos?

import concurrent.futures
import multiprocessing
from scqbf.scqbf_evaluator import *
from scqbf.scqbf_solution import *
from scqbf.scqbf_ga import *
from scqbf.scqbf_instance import *

import pandas as pd
import numpy as np
import pickle
import random

def run_experiment(config: dict, pop_size, mutation_rate) -> pd.DataFrame:
    results = []
    exp_num = 0

    instance_paths = [(f"instances/gen{i}/instance{j}.txt", i, j) for i in range(1, 4) for j in range(1, 6)]

    for instance_path, gen, inst in instance_paths:
        # Read Instance
        instance = read_max_sc_qbf_instance(instance_path)
        print(f"{exp_num}: {inst}th instance of generation strategy {gen}. Path: {instance_path}")
        exp_num += 1
        
        # Set stop criteria values
        if config.get('stop_criteria', 'time') == 'time':
            time_limit = 60*30 # 30 minutes
            n_generations = None
        elif config.get('stop_criteria', 'time') == 'generations':
            time_limit = None
            n_generations = 1000

        # Run Genetic Algorithm
        ga = ScQbfGeneticAlgorithm(
            instance, 
            time_limit_secs=time_limit, 
            generations=n_generations, 
            pop_size=pop_size, 
            mutation_rate=mutation_rate, 
            debug=True, 
            config=config
        )
        best_solution = ga.solve()
        
        # Evaluate and save metrics
        evaluator = ScQbfEvaluator(instance)
        results.append({
            'gen': gen,
            'inst': inst,
            'n': instance.n,
            'best_objective': round(evaluator.evaluate_objfun(best_solution), 4),
            'coverage': evaluator.evaluate_coverage(best_solution),
            'time_taken': round(ga.solve_time),
            'n_generations': ga.current_generation,
        })
        
        # Print values
        last_result = results[-1]
        print(f"\tBest objective value: {last_result['best_objective']:.4f}")
        print(f"Selected elements: {best_solution.elements}")
        print(f"\tCoverage: {last_result['coverage']:.2%}")
        print(f"\tGenerations executed: {last_result['n_generations']}")
        print()

    df = pd.DataFrame(results)
    return df

def run_and_save_experiment(args):
    """Wrapper to run one experiment and save its results."""
    config, pop_size, mutation_rate, exp_id = args
    results_df = run_experiment(config, pop_size, mutation_rate)
    filename = f'results/experiment_results_{exp_id}.csv'
    results_df.to_csv(filename, index=False)
    
    return exp_id, filename


def run_all(experiments, max_workers=2):
    ctx = multiprocessing.get_context('spawn')
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(run_and_save_experiment, args) for args in experiments]
        for fut in concurrent.futures.as_completed(futures):
            exp_id, filename = fut.result()  # exceções aqui serão lançadas
            print(f"Experimento {exp_id} salvo em {filename}")
            results.append(pd.read_csv(filename))
            
    return pd.concat(results, ignore_index=True)

if __name__ == '__main__':
    experiments = [
        ({'stop_criteria': 'time'}, 100, 1/100, 1),
        ({'stop_criteria': 'time', 'crossover_type': 'uniform'}, 100, 1/100, 2),
    ]
    combined = run_all(experiments, max_workers=2)
    combined.to_csv('results/all_experiments.csv', index=False)
    print("Todos finalizados.")