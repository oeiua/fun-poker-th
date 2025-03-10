"""
Memory usage tracking for leak detection.
"""
import os
import psutil
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
import time

def track_memory_usage() -> Dict[str, Any]:
    """
    Track current memory usage.
    
    Returns:
        Dictionary with memory usage statistics
    """
    # Get memory info from psutil
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # Get system memory info
    system_mem = psutil.virtual_memory()
    
    # Get GPU memory info if available
    gpu_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem[i] = {
                'allocated': torch.cuda.memory_allocated(i) / (1024 ** 2),  # MB
                'reserved': torch.cuda.memory_reserved(i) / (1024 ** 2),  # MB
                'max_allocated': torch.cuda.max_memory_allocated(i) / (1024 ** 2)  # MB
            }
    
    # Compile memory usage information
    memory_usage = {
        'timestamp': time.time(),
        'rss': mem_info.rss,  # Resident Set Size
        'vms': mem_info.vms,  # Virtual Memory Size
        'used': system_mem.used,
        'total': system_mem.total,
        'percent': system_mem.percent,
        'gpu': gpu_mem
    }
    
    return memory_usage

def detect_memory_leaks(
    memory_history: List[Dict[str, Any]],
    threshold: float = 10.0,
    window_size: int = 5
) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect potential memory leaks from history.
    
    Args:
        memory_history: List of memory usage snapshots
        threshold: Percentage increase threshold to trigger a warning
        window_size: Number of snapshots to analyze for trend
        
    Returns:
        Tuple of (leak_detected, leak_info)
    """
    if len(memory_history) < window_size:
        return False, {}
    
    # Extract recent history
    recent_history = memory_history[-window_size:]
    
    # Calculate RSS growth trend
    rss_values = [snapshot['rss'] for snapshot in recent_history]
    timestamps = [snapshot['timestamp'] for snapshot in recent_history]
    
    # Convert to numpy arrays
    rss_array = np.array(rss_values)
    timestamps_array = np.array(timestamps)
    
    # Calculate linear regression
    try:
        # Normalize x values for numerical stability
        x = timestamps_array - timestamps_array[0]
        y = rss_array
        
        if len(x) == 0 or np.all(x == 0):
            return False, {}
        
        # Calculate slope and intercept
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate growth percentage
        start_value = rss_array[0]
        end_value = rss_array[-1]
        growth_percent = (end_value - start_value) / max(1, start_value) * 100
        
        # Check for potential leak
        leak_detected = growth_percent > threshold and slope > 0
        
        leak_info = {
            'growth_percent': growth_percent,
            'slope': slope,
            'start_value': start_value,
            'end_value': end_value,
            'window_size': window_size,
            'time_span': timestamps_array[-1] - timestamps_array[0]
        }
        
        return leak_detected, leak_info
    
    except Exception as e:
        # In case of any calculation error
        return False, {'error': str(e)}

def plot_memory_usage(
    memory_history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot memory usage over time.
    
    Args:
        memory_history: List of memory usage snapshots
        save_path: Path to save the plot
        figsize: Figure size
    """
    if not memory_history:
        return
    
    plt.figure(figsize=figsize)
    
    # Extract data
    timestamps = [snapshot['timestamp'] for snapshot in memory_history]
    # Convert to relative time in minutes
    relative_times = [(t - timestamps[0]) / 60 for t in timestamps]
    
    rss_values = [snapshot['rss'] / (1024 * 1024) for snapshot in memory_history]  # Convert to MB
    vms_values = [snapshot['vms'] / (1024 * 1024) for snapshot in memory_history]  # Convert to MB
    system_percent = [snapshot['percent'] for snapshot in memory_history]
    
    # Plot RSS and VMS
    plt.subplot(2, 1, 1)
    plt.plot(relative_times, rss_values, label='RSS (MB)')
    plt.plot(relative_times, vms_values, label='VMS (MB)')
    plt.title('Process Memory Usage')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Memory (MB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot system memory percentage
    plt.subplot(2, 1, 2)
    plt.plot(relative_times, system_percent, label='System Memory %', color='red')
    plt.title('System Memory Usage')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Usage (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot GPU memory if available
    if memory_history[0]['gpu'] and torch.cuda.is_available():
        plt.figure(figsize=figsize)
        
        # Get number of GPUs
        num_gpus = len(memory_history[0]['gpu'])
        
        for gpu_idx in range(num_gpus):
            # Extract GPU data
            allocated = [snapshot['gpu'].get(gpu_idx, {}).get('allocated', 0) for snapshot in memory_history]
            reserved = [snapshot['gpu'].get(gpu_idx, {}).get('reserved', 0) for snapshot in memory_history]
            
            plt.subplot(num_gpus, 1, gpu_idx + 1)
            plt.plot(relative_times, allocated, label='Allocated (MB)')
            plt.plot(relative_times, reserved, label='Reserved (MB)')
            plt.title(f'GPU {gpu_idx} Memory Usage')
            plt.xlabel('Time (minutes)')
            plt.ylabel('Memory (MB)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save GPU plot if requested
        if save_path:
            gpu_save_path = save_path.replace('.png', '_gpu.png')
            plt.savefig(gpu_save_path, dpi=300, bbox_inches='tight')
    
    # Save or show CPU/system memory plot
    if save_path:
        plt.figure(1)  # Switch back to first figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    
    # Close plots to free memory
    plt.close('all')

def force_gc() -> Dict[str, Any]:
    """
    Force garbage collection and return memory stats before and after.
    
    Returns:
        Dictionary with before/after memory statistics
    """
    # Get memory before collection
    before = track_memory_usage()
    
    # Force Python garbage collection
    collected = gc.collect()
    
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get memory after collection
    after = track_memory_usage()
    
    # Calculate differences
    rss_diff = before['rss'] - after['rss']
    vms_diff = before['vms'] - after['vms']
    percent_diff = before['percent'] - after['percent']
    
    return {
        'before': before,
        'after': after,
        'collected_objects': collected,
        'rss_freed': rss_diff,
        'vms_freed': vms_diff,
        'percent_freed': percent_diff
    }