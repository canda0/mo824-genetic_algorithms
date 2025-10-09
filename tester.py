# Versão que cria um processo por GA. Talvez paralelizar run_experiment para usar mais núcleos?

import concurrent.futures
import multiprocessing
from joblib import Parallel, delayed
import pandas as pd
import time

from scqbf.scqbf_evaluator import *
from scqbf.scqbf_solution import *
from scqbf.scqbf_ga import *
from scqbf.scqbf_instance import *


def run_experiment(config: dict, pop_size, mutation_rate) -> pd.DataFrame:
    def process_instance(instance_path, gen, inst):
        # Read Instance
        instance = read_max_sc_qbf_instance(instance_path)

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
            debug=False, 
            config=config
        )
        best_solution = ga.solve()
        
        # Evaluate and save metrics
        evaluator = ScQbfEvaluator(instance)
        results = {
            'gen': gen,
            'inst': inst,
            'n': instance.n,
            'pop_size': pop_size,
            'mutation_rate': round(mutation_rate, 4),
            'best_objective': round(evaluator.evaluate_objfun(best_solution), 2),
            'coverage': evaluator.evaluate_coverage(best_solution),
            'time_taken': round(ga.solve_time),
            'n_generations': ga.current_generation,
        }

        # Print values
        print(f"Results for instance {inst} of generation strategy {gen} with {results['n']} variables:")
        print(f"---- Best objective value: {results['best_objective']:.2f}")
        print(f"---- Selected elements: {best_solution.elements}")
        print(f"---- Coverage: {results['coverage']:.2%}")
        print(f"---- Generations executed: {results['n_generations']}")
        print(f"---- Time taken (s): {results['time_taken']}")
        print()

        return results
    
    # Define instances to process
    instance_paths = [(f"instances/gen{i}/instance{j}.txt", i, j) for i in range(1, 4) for j in range(1, 6)]

    # Parrallel processing of instances
    results = Parallel(n_jobs=4)(
        delayed(process_instance)(instance_path, gen, inst)
        for instance_path, gen, inst in instance_paths
    )
    df = pd.DataFrame(results)

    return df

def run_and_save_experiment(args):
    """Wrapper to run one experiment and save its results."""
    config, pop_size, mutation_rate, exp_id = args
    results_df = run_experiment(config, pop_size, mutation_rate)
    results_df['experiment_id'] = exp_id
    filename = f'results/experiment_results_{exp_id}.csv'
    results_df.to_csv(filename, index=False)
    
    return exp_id, filename


def run_all(experiments, max_workers=2):
    # Create multiprocessing context
    ctx = multiprocessing.get_context('spawn')
    results = []

    # Create a process pool with 'max_workers' and run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(run_and_save_experiment, args) for args in experiments]
        for fut in concurrent.futures.as_completed(futures):
            exp_id, filename = fut.result()  # exceções aqui serão lançadas
            print(f"Experimento {exp_id} salvo em {filename}")
            results.append(pd.read_csv(filename))
            
    return pd.concat(results, ignore_index=True)

if __name__ == '__main__':
    # Define experiments as tuples of (config, pop_size, mutation_rate, exp_id)
    experiments = [
        ({'stop_criteria': 'time'}, 100, 1/100, 1),
        ({'stop_criteria': 'time'}, 300, 1/100, 2),
        ({'stop_criteria': 'time'}, 100, 1/50, 3),
        ({'stop_criteria': 'time', 'crossover_type': 'uniform'}, 100, 1/100, 4),
        ({'stop_criteria': 'time', 'evolution_mode': 'steady_state'}, 100, 1/100, 5),
    ]

    print("Starting experiments with the following configurations:")
    for config, pop_size, mutation_rate, exp_id in experiments:
        print(f" - Experiment {exp_id}: {config}, Pop Size: {pop_size}, Mutation Rate: {mutation_rate}")
    
    # Run all experiments
    start_time = time.perf_counter()
    combined = run_all(experiments, max_workers=2)
    combined.to_csv('results/all_experiments.csv', index=False)
    end_time = time.perf_counter() - start_time
    print(f"All experiments completed in {end_time:.2f} seconds.")