import sys
import numpy as np
import jax
import jax.numpy as jnp
import h5py

from stokeslet_solver_jax import compute_flow_field_grid 

if __name__ == "__main__":
    print("Loading input data...")
    all_parameters = jnp.array(np.loadtxt('all_parameters.txt'))
    coeff_time = jnp.array(np.loadtxt('coeff_time.txt'))
    
    # 1. GENERATE THE SPATIAL GRID
    x_vals = np.linspace(-10, 10, 200)
    y_vals = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = jnp.array(np.stack((X, Y), axis=-1))
    
    # 2. GENERATE THE TIME ARRAY
    t_start = 0.0
    t_end = 3.0
    time_step = 5e-3
    time_array_cpu = np.arange(t_start, t_end + 1e-10, time_step) 
    
    Nt = len(time_array_cpu)
    
    # ---------------------------------------------------------
    # NEW: COMMAND LINE CHUNK LOGIC
    # ---------------------------------------------------------
    # Read chunk index and total chunks from command line arguments
    # Usage: python run_sim.py <chunk_index> <total_chunks>
    if len(sys.argv) == 3:
        array_idx_str = sys.argv[1]
        total_jobs = int(sys.argv[2])
    else:
        # Fallback if run without arguments
        array_idx_str = '1'
        total_jobs = 4 
        
    array_idx = int(array_idx_str) - 1 # Python is 0-indexed
    all_indices = np.arange(Nt)
    
    # ---------------------------------------------------------
    # CUSTOM UNEQUAL CHUNKING FOR GPU LOAD BALANCING
    # ---------------------------------------------------------
    if total_jobs == 4:
        # Hardcoded bounds to balance compute time 
        # (Later timesteps require more integration history)
        custom_chunks = [
            all_indices[0:300],   # Chunk 1: indices 0 to 299
            all_indices[300:450], # Chunk 2: indices 300 to 449
            all_indices[450:526], # Chunk 3: indices 450 to 525
            all_indices[526:]     # Chunk 4: indices 526 to the end
        ]
        my_chunk = custom_chunks[array_idx]
    else:
        # Fallback to equal splits if you ever change the number of GPUs
        chunks = np.array_split(all_indices, total_jobs)
        my_chunk = chunks[array_idx]
    
    start_idx = my_chunk[0]
    end_idx = my_chunk[-1] + 1
    Nt_chunk = len(my_chunk)
    
    print("-" * 40)
    print(f"Chunk Index: {array_idx_str} / {total_jobs}")
    print(f"Computing global time steps {start_idx} to {end_idx - 1}")
    print(f"Total steps for this specific GPU: {Nt_chunk}")
    print("-" * 40)
    
    # Pre-allocate ONLY for this chunk's steps
    velocity_4D_array_cpu = np.zeros((Nt_chunk, 200, 200, 2))
    
    # ---------------------------------------------------------
    # 3. BATCHED GPU COMPUTATION
    # ---------------------------------------------------------
    print("Dispatching computation to the GPU...")
    
    # Use enumerate to get a local index for filling our smaller CPU array
    for local_idx, global_t_idx in enumerate(my_chunk):
        current_t = jnp.array(time_array_cpu[global_t_idx])
        
        grid_velocities_gpu = compute_flow_field_grid(
            target_grid=grid_points,
            all_parameters=all_parameters,
            current_t=current_t,
            coeff_time=coeff_time,
            dt_int=2.5e-6
        )

        velocity_4D_array_cpu[local_idx] = np.array(grid_velocities_gpu.block_until_ready())
        
        if local_idx % 10 == 0 or local_idx == Nt_chunk - 1:
            print(f"Completed local step {local_idx + 1}/{Nt_chunk} (Global Step: {global_t_idx})")

    # ---------------------------------------------------------
    # 4. FORMAT AND EXPORT TO H5PY
    # ---------------------------------------------------------
    # Save with a zero-padded suffix (e.g., part_01.h5, part_04.h5)
    output_filename = f'single_sperm_flow_field_part_{array_idx_str.zfill(2)}.h5'
    print(f"Formatting data for h5py export to {output_filename}...")
    
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('time', data=time_array_cpu)
        f.create_dataset('x_coords', data=x_vals)
        f.create_dataset('y_coords', data=y_vals)
        f.create_dataset('velocity_field', data=velocity_4D_array_cpu, compression='gzip')
    
    print(f"Data saved successfully to {output_filename}")