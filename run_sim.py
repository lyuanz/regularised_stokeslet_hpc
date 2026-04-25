import numpy as np
import jax
import jax.numpy as jnp
import h5py

# Ensure this imports your newly created JAX version!
from stokeslet_solver_jax import compute_flow_field_over_time 

if __name__ == "__main__":
    print("Loading input data...")
    # 1. Read from hard drive using CPU, then beam to GPU using jnp.array
    all_parameters = jnp.array(np.loadtxt('all_parameters.txt'))
    coeff_time = jnp.array(np.loadtxt('coeff_time.txt'))
    window_map = jnp.array(np.loadtxt('TimeWindow.txt'))
    
    # 2. GENERATE THE SPATIAL GRID
    # Build on CPU first, then transfer to GPU
    x_vals = np.linspace(-10, 10, 200)
    y_vals = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = jnp.array(np.stack((X, Y), axis=-1))
    
    # 3. GENERATE THE TIME ARRAY
    t_start = 0.0
    t_end = 3.0
    time_step = 5e-3
    
    time_array_cpu = np.arange(t_start, t_end + 1e-10, time_step) 
    time_array = jnp.array(time_array_cpu)
    
    print(f"Total time steps to compute: {len(time_array)}")
    
    # 4. RUN THE MASSIVELY PARALLEL COMPUTATION
    print("Dispatching computation to the GPU...")
    velocity_4D_array_gpu = compute_flow_field_over_time(
        target_grid=grid_points,
        all_parameters=all_parameters,
        time_array=time_array,
        coeff_time=coeff_time,
        window_map=window_map
    )

    # CRITICAL HPC ADDITION: Block until ready
    # JAX is asynchronous. The Python interpreter will instantly move to the next 
    # line before the GPU is finished unless we explicitly force it to wait.
    velocity_4D_array_gpu = velocity_4D_array_gpu.block_until_ready()

    print(f"Calculation complete! Array shape: {velocity_4D_array_gpu.shape}")
    
    # ---------------------------------------------------------
    # 5. FORMAT AND EXPORT TO H5PY
    # ---------------------------------------------------------
    print("Formatting data for h5py export...")
    
    # CRITICAL HPC ADDITION: Transfer back to CPU memory
    # h5py cannot read directly from GPU VRAM. We must pull the final array 
    # back to the system RAM using standard np.array()
    velocity_4D_array_cpu = np.array(velocity_4D_array_gpu)
    
    # Save the data
    output_filename = 'single_sperm_flow_field.h5'
    with h5py.File(output_filename, 'w') as f:
        # We use the CPU versions of these arrays for saving
        f.create_dataset('time', data=time_array_cpu)
        f.create_dataset('x_coords', data=x_vals)
        f.create_dataset('y_coords', data=y_vals)
        
        # Save the massive 4D array with GZIP compression
        f.create_dataset('velocity_field', data=velocity_4D_array_cpu, compression='gzip')
    
    print(f"Data saved to {output_filename}")