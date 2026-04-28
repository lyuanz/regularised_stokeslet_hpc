import numpy as np
import jax.numpy as jnp

from stokeslet_solver_jax import compute_flow_field_grid

if __name__ == "__main__":
    print("Loading input data...")
    all_parameters = jnp.array(np.loadtxt('all_parameters.txt'))
    coeff_time = jnp.array(np.loadtxt('coeff_time.txt'))
    
    # GENERATE THE SPATIAL GRID
    x_vals = np.linspace(-10, 10, 200)
    y_vals = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    grid_points = jnp.array(np.stack((X, Y), axis=-1))
    
    # 1. Choose the worst-case snapshot to test
    t_test = jnp.array(3)
    
    print("Calculating Ground Truth (dt_int = 5e-7)...")
    v_true = compute_flow_field_grid(
        target_grid=grid_points, 
        all_parameters=all_parameters, 
        current_t=t_test, 
        coeff_time=coeff_time,
        dt_int=5e-7,  # The absurdly small ground truth
        dt=5e-3
    ).block_until_ready()
    true_norm = jnp.linalg.norm(v_true)
    
    # 2. Define the step sizes to test
    dt_test_list = [5e-3 / (2**i) for i in range(14)]
    errors = []
    
    print("Running Convergence Sweep...")
    for dt in dt_test_list:
        v_test = compute_flow_field_grid(
            target_grid=grid_points, 
            all_parameters=all_parameters, 
            current_t=t_test, 
            coeff_time=coeff_time,
            dt_int=dt,
            dt=5e-3
        ).block_until_ready()
        
        # 3. Calculate Relative L2 Error
        diff_norm = jnp.linalg.norm(v_test - v_true)
        error = float(diff_norm / true_norm)
        
        errors.append(error)
        print(f"dt_int: {dt:.6f} | Relative Error: {error:.6e}")
        
    # ---------------------------------------------------------
    # NEW: EXPORT DATA TO TXT
    # ---------------------------------------------------------
    print("Exporting results to convergence_data.txt...")
    
    # Zip the two 1D lists together into a single 2D array (Nx2 shape)
    export_data = np.column_stack((dt_test_list, errors))
    
    # Save the array to a text file
    np.savetxt(
        "convergence_data.txt", 
        export_data, 
        delimiter="\t",              # Separate columns with a tab
        fmt="%.8e",                  # Format as scientific notation with 8 decimals
        header="dt_int\tL2_error",   # Add a clean header row
        comments=""                  # Prevent NumPy from adding a '#' before the header
    )
    
    print("Export complete! You can now download convergence_data.txt to your laptop.")