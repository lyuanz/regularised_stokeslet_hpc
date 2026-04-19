import numpy as np
import h5py
from stokeslet_solver import compute_flow_field_over_time

if __name__ == "__main__":
    print("Loading input data...")
    all_parameters = np.loadtxt('all_parameters.txt')
    coeff_time = np.loadtxt('coeff_time.txt')
    window_map = np.loadtxt('TImeWindow.txt')
    
    # 1. GENERATE THE SPATIAL GRID (as before)
    x_vals = np.linspace(-10, 10, 200)
    y_vals = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = np.stack((X, Y), axis=-1)
    
    # 2. GENERATE THE TIME ARRAY
    t_start = 0.0
    t_end = 3.0
    time_step = 5e-3
    
    # np.arange(start, stop, step). 
    # We add a tiny bit to t_end to ensure 3.0 is included despite floating point math.
    time_array = np.arange(t_start, t_end + 1e-10, time_step) 
    
    print(f"Total time steps to compute: {len(time_array)}")
    
    # 3. RUN THE PARALLEL COMPUTATION
    velocity_4D_array = compute_flow_field_over_time(
        target_grid=grid_points,
        all_parameters=all_parameters,
        time_array=time_array,
        coeff_time=coeff_time,
        window_map=window_map
    )

    print(f"Calculation complete! Array shape: {velocity_4D_array.shape}")
    
    # ---------------------------------------------------------
    # 4. FORMAT AND EXPORT TO H5PY
    # ---------------------------------------------------------
    print("Formatting data for h5py export...")
    
    # Save the data
    output_filename = 'single_sperm_flow_field.h5'
    with h5py.File(output_filename, 'w') as f:
        # We can save the grid coordinates and time array too!
        f.create_dataset('time', data=time_array)
        f.create_dataset('x_coords', data=x_vals)
        f.create_dataset('y_coords', data=y_vals)
        
        # Save the massive 4D array with GZIP compression
        f.create_dataset('velocity_field', data=velocity_4D_array, compression='gzip')
    
    print(f"Data saved to {output_filename}")