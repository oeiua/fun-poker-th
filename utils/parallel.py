"""
Parallel processing utilities for efficient training.
"""
import os
import torch
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import time
import numpy as np
from tqdm import tqdm

def parallel_map(
    func: Callable,
    items: List[Any],
    num_workers: Optional[int] = None,
    use_tqdm: bool = True,
    desc: str = "Processing"
) -> List[Any]:
    """
    Apply a function to a list of items in parallel.
    
    Args:
        func: Function to apply
        items: List of items to process
        num_workers: Number of worker processes (default: CPU count - 1)
        use_tqdm: Whether to show progress bar
        desc: Description for progress bar
        
    Returns:
        List of results
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    
    # If only one worker or very few items, run sequentially
    if num_workers == 1 or len(items) <= 1:
        if use_tqdm:
            return [func(item) for item in tqdm(items, desc=desc)]
        else:
            return [func(item) for item in items]
    
    # Run in parallel
    with mp.Pool(processes=num_workers) as pool:
        if use_tqdm:
            results = list(tqdm(pool.imap(func, items), total=len(items), desc=desc))
        else:
            results = pool.map(func, items)
    
    return results

def parallel_starmap(
    func: Callable,
    args_list: List[Tuple],
    num_workers: Optional[int] = None,
    use_tqdm: bool = True,
    desc: str = "Processing"
) -> List[Any]:
    """
    Apply a function to a list of argument tuples in parallel.
    
    Args:
        func: Function to apply
        args_list: List of argument tuples for the function
        num_workers: Number of worker processes (default: CPU count - 1)
        use_tqdm: Whether to show progress bar
        desc: Description for progress bar
        
    Returns:
        List of results
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    
    # If only one worker or very few items, run sequentially
    if num_workers == 1 or len(args_list) <= 1:
        if use_tqdm:
            return [func(*args) for args in tqdm(args_list, desc=desc)]
        else:
            return [func(*args) for args in args_list]
    
    # Run in parallel
    with mp.Pool(processes=num_workers) as pool:
        if use_tqdm:
            results = list(tqdm(pool.istarmap(func, args_list), total=len(args_list), desc=desc))
        else:
            results = pool.starmap(func, args_list)
    
    return results

def batch_process(
    func: Callable,
    items: List[Any],
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    use_tqdm: bool = True,
    desc: str = "Processing batches"
) -> List[Any]:
    """
    Process items in batches to reduce memory usage.
    
    Args:
        func: Function that takes a batch of items and returns a list of results
        items: List of items to process
        batch_size: Size of each batch
        num_workers: Number of worker processes for parallel batch processing
        use_tqdm: Whether to show progress bar
        desc: Description for progress bar
        
    Returns:
        List of results (concatenated from all batches)
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    
    # Split items into batches
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    # Process batches (optionally in parallel)
    if num_workers > 1 and len(batches) > 1:
        # Process batches in parallel
        batch_results = parallel_map(func, batches, num_workers, use_tqdm, desc)
    else:
        # Process batches sequentially
        if use_tqdm:
            batch_results = [func(batch) for batch in tqdm(batches, desc=desc)]
        else:
            batch_results = [func(batch) for batch in batches]
    
    # Concatenate results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    return all_results

def distribute_tasks(
    items: List[Any],
    num_workers: int
) -> List[List[Any]]:
    """
    Distribute tasks among workers for manual parallelization.
    
    Args:
        items: List of items to distribute
        num_workers: Number of workers
        
    Returns:
        List of task lists, one per worker
    """
    num_workers = min(num_workers, len(items))
    worker_tasks = [[] for _ in range(num_workers)]
    
    # Distribute tasks
    for i, item in enumerate(items):
        worker_idx = i % num_workers
        worker_tasks[worker_idx].append(item)
    
    return worker_tasks

def parallel_evaluation(
    evaluate_func: Callable,
    agents: List[Any],
    num_workers: int,
    games_per_eval: int
) -> List[float]:
    """
    Evaluate a list of agents in parallel.
    
    Args:
        evaluate_func: Function that takes (agent, num_games) and returns fitness
        agents: List of agents to evaluate
        num_workers: Number of worker processes
        games_per_eval: Number of games per evaluation
        
    Returns:
        List of fitness scores
    """
    # Create argument tuples for each agent
    args_list = [(agent, games_per_eval) for agent in agents]
    
    # Run evaluations in parallel
    return parallel_starmap(
        evaluate_func,
        args_list,
        num_workers=num_workers,
        use_tqdm=True,
        desc="Evaluating agents"
    )

# Custom version of Pool.istarmap (not provided in standard multiprocessing)
def _istarmap_wrapper(args):
    return args[0](*args[1])

# Monkey patch Pool if istarmap is not available
if not hasattr(mp.pool.Pool, 'istarmap'):
    mp.pool.Pool.istarmap = lambda self, func, iterable, chunksize=1: self.imap(
        _istarmap_wrapper, ((func, args) for args in iterable), chunksize
    )