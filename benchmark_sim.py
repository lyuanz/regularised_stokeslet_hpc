import numpy as np
import jax
import jax.numpy as jnp
import time

from stokeslet_solver_jax import compute_flow_field_grid 

if __name__ == "__main__":
    print("Loading input data...")
    all_parameters = jnp.array(np.loadtxt('all_parameters.txt'))
    coeff_time = jnp.array(np.loadtxt('coeff_time.txt'))
    window_map = jnp.array(np.loadtxt('TimeWindow.txt'))
    
    # 1. GENERATE THE SPATIAL GRID
    x_vals = np.linspace(-10, 10, 200)
    y_vals = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = jnp.array(np.stack((X, Y), axis=-1))
    
    print("Running JAX compilation step...")
    t_comp_start = time.perf_counter()
    _ = compute_flow_field_grid(
        target_grid=grid_points, all_parameters=all_parameters,
        current_t=jnp.array(0.0), coeff_time=coeff_time,
        window_map=window_map, dt_int=2.5e-6
    ).block_until_ready()
    t_comp_end = time.perf_counter()
    print(f"Compilation took {t_comp_end - t_comp_start:.2f} seconds")
    
    print("Running Maximum Workload Benchmark...")
    t_heavy_start = time.perf_counter()
    # Force the calculation at the very end of the simulation
    _ = compute_flow_field_grid(
        target_grid=grid_points, all_parameters=all_parameters,
        current_t=jnp.array(3.0), # <--- MAX TIME
        coeff_time=coeff_time, window_map=window_map, dt_int=2.5e-6
    ).block_until_ready()
    t_heavy_end = time.perf_counter()
    
    max_step_time = t_heavy_end - t_heavy_start
    print(f"Heaviest step (t=3.0) took {max_step_time:.4f} seconds")