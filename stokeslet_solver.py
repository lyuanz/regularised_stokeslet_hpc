import numpy as np
import math
from numba import njit, prange

# ---------------------------------------------------------
# 1. CORE MATH: STOKESLET TENSOR
# ---------------------------------------------------------
@njit # jit, no-python
def calc_stokeslet_tensor(r_vec, 
                          epsilon,
                          delta_t, 
                          nu=168
                         ):
    
    """
    Computes the 2x2 unsteady regularised Stokeslet tensor R(r, t-tau).

    Parameters
    ----------
    r_vec -> np.ndarray : (x, y) displacement vector of the point of interest with respect to the centre of regularised Stokeslet
    epsilon -> float : radius of regularised Stokeslet
    delta_t -> float : Duration from force pulse application
    nu -> int : kinematic viscosity

    Returns
    -------
    R(r, t-tau) -> np.ndarray : unsteady regularised Stokeslet tensor at position r from the source, at time t-tau
    """
    
    r = np.linalg.norm(r_vec)
    zeta = math.sqrt(nu * delta_t + 0.5 * epsilon**2)
    
    # Precompute common terms
    zeta2 = zeta**2
    zeta3 = zeta**3
    sqrt_pi = math.sqrt(math.pi)
    
    exp_term = math.exp(-(r**2) / (4 * zeta2))
    erf_term = math.erf(r / (2 * zeta))
    
    # Handle r -> 0 numerically to avoid division by zero
    if r < 1e-12:
        # Taylor expansion limit of the tensor as r approaches 0
        limit_val = 1/ (3 * sqrt_pi * zeta3)
        return np.eye(2) * limit_val

    r2 = r**2
    r3 = r**3
    
    # Term A (Coefficient for Identity Matrix)
    A = (1 / (2 * sqrt_pi * zeta3)) * exp_term \
      - (1 / r3) * erf_term \
      + (1 / (sqrt_pi * zeta * r2)) * exp_term
      
    # Term B (Coefficient for r_hat outer product r_hat)
    B = (3 / r3) * erf_term \
      - (3 / (sqrt_pi * zeta * r2)) * exp_term \
      - (1 / (2 * sqrt_pi * zeta3)) * exp_term
      
    r_hat = r_vec / r
    # Outer product for r_hat * r_hat
    rr_tensor = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            rr_tensor[i, j] = r_hat[i] * r_hat[j]
            
    # Assemble the 2x2 tensor
    tensor = np.eye(2) * A + rr_tensor * B
    return tensor
    
# ---------------------------------------------------------
# 2. TIME WINDOW LOGIC
# ---------------------------------------------------------
@njit
def get_time_window(r, 
                    window_map
                   ):
    """
    Looks up the time window for a given distance r.

    Parameters
    ----------
    r -> float : distance between position of interest and the source
    window_map -> np.ndarray : An array recording the range of r values and the time window of each range, i.e. [[r_min, r_max, time_window], [...]]

    Returns
    -------
    time_window -> float : The time window
    """
    for i in range(window_map.shape[0]):
        if window_map[i, 0] <= r < window_map[i, 1]:
            return window_map[i, 2]
            
    return window_map[-1, 2]
    
# ---------------------------------------------------------
# 3. CONVOLUTION INTEGRATION
# ---------------------------------------------------------
@njit
def compute_flow_velocity(target_pos, 
                            regStokes_pos, 
                            current_t, 
                            base_force, 
                            multiplier_array,  
                            window_map, 
                            epsilon, 
                            dt=5e-3,
                            nu=168, 
                            rho=1/168
                            ):
    """
    Computes the flow velocity vector u at target_pos due to a regularised Stokeslet.
    Performs trapezoidal numerical integration over the valid time window.

    Parameters
    ----------
    target_pos -> np.ndarray : (x, y) coordinates of the target position
    regStokes_pos -> np.ndarray : (x, y) coordinates of the origin of regularised Stokeslet
    current_t -> float : current timepoint
    base_force -> np.ndarray : base force of the regularised Stokeslet
    multiplier_array -> np.ndarray : 1D array of scalar multipliers for base_force over one flagellar beat period
    window_map -> float : time window beyond which the convolution integral will be truncated
    epsilon -> float : radius of the regularised Stokeslet
    dt -> float : timestep
    nu -> int : kinematic viscosity
    rho -> float : density

    Returns
    -------
    np.ndarray : Fluid flow velocity at the position of interest
    """
    r_vec = target_pos - regStokes_pos
    r = np.linalg.norm(r_vec)
    
    window = get_time_window(r, window_map)
    t_start = max(0.0, current_t - window)
    
    idx_start = int(round(t_start / dt))
    idx_end = int(round(current_t / dt))
        
    N_period = len(multiplier_array)
    T_period = N_period * dt
    integral_sum = np.zeros(2, dtype=np.float64)
    
    # 5 steps is a safe danger zone, but you can dynamically scale this later if needed
    danger_zone_steps = 5
    idx_danger = max(idx_start, idx_end - danger_zone_steps)
    
    # ---------------------------------------------------------
    # PART 1: BULK INTEGRATION (Standard Trapezoidal Rule)
    # ---------------------------------------------------------
    for i in range(idx_start, idx_danger):
        tau1 = i * dt
        tau2 = (i + 1) * dt
        
        wrap_idx1 = i % N_period
        wrap_idx2 = (i + 1) % N_period
        
        dt1 = current_t - tau1
        R1 = calc_stokeslet_tensor(r_vec, epsilon, dt1, nu)
        F1 = base_force * multiplier_array[wrap_idx1]
        integrand1 = R1 @ F1
        
        dt2 = current_t - tau2
        R2 = calc_stokeslet_tensor(r_vec, epsilon, dt2, nu)
        F2 = base_force * multiplier_array[wrap_idx2]
        integrand2 = R2 @ F2
        
        integral_sum += 0.5 * (integrand1 + integrand2) * dt

    # ---------------------------------------------------------
    # PART 2: DANGER ZONE (Cubic Clustered Integration)
    # ---------------------------------------------------------
    # Calculate the exact time gap remaining for the danger zone
    tau_danger = idx_danger * dt
    S_max = current_t - tau_danger
    
    if S_max > 0:
        N_micro = 50  # Number of clustered points
        
        # We integrate over s (age of the force), where s = current_t - tau
        for j in range(N_micro):
            # CUBIC clustering packs points heavily near s=0
            s_1 = S_max * (j / N_micro)**3
            s_2 = S_max * ((j + 1) / N_micro)**3
            ds = s_2 - s_1
            
            tau1 = current_t - s_1
            tau2 = current_t - s_2
            
            # --- Continuous Interpolation for F1 at tau1 ---
            idx_float1 = (tau1 % T_period) / dt
            idx_low1 = int(idx_float1)
            idx_high1 = (idx_low1 + 1) % N_period
            w1 = idx_float1 - int(idx_float1)
            F1 = base_force * (multiplier_array[idx_low1] * (1.0 - w1) + multiplier_array[idx_high1] * w1)
            
            # --- Continuous Interpolation for F2 at tau2 ---
            idx_float2 = (tau2 % T_period) / dt
            idx_low2 = int(idx_float2)
            idx_high2 = (idx_low2 + 1) % N_period
            w2 = idx_float2 - int(idx_float2)
            F2 = base_force * (multiplier_array[idx_low2] * (1.0 - w2) + multiplier_array[idx_high2] * w2)
            
            # Evaluate the tensors
            R1 = calc_stokeslet_tensor(r_vec, epsilon, s_1, nu)
            R2 = calc_stokeslet_tensor(r_vec, epsilon, s_2, nu)
            
            integrand1 = R1 @ F1
            integrand2 = R2 @ F2
            
            integral_sum += 0.5 * (integrand1 + integrand2) * ds
            
    prefactor = 1.0 / (4.0 * math.pi * rho)
    return prefactor * integral_sum
    
# ---------------------------------------------------------
# MULTI-STOKESLET SUPERPOSITION
# ---------------------------------------------------------
@njit
def compute_total_velocity(target_pos, 
                           all_parameters, 
                           current_t,  
                           coeff_time,
                           window_map, 
                           dt=5e-3,
                           nu=168, 
                           rho=1/168
                          ) -> np.ndarray:
    """
    Calculates the combined flow field from all regularised Stokeslets of the same cell 
    using the principle of linear superposition.

    Parameters
    ----------
    target_pos -> np.ndarray : (x, y) coordinates of the target position
    all_parameters -> np.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    current_t -> float : current timepoint
    coeff_time -> np.ndarray : Scalar multipliers for force over one flagellar beat period
    window_map -> np.ndarray : time window beyond which the convolution integral will be truncated
    dt -> float : timestep
    nu -> int : kinematic viscosity
    rho -> float : density

    Returns
    -------
    np.ndarray : Fluid flow velocity at the position of interest due to all the Stokeslets of a sperm cell
    """
    total_u = np.zeros(2)
    num_stokeslets = all_parameters.shape[0]
    stokeslet_positions = all_parameters[:, :2]
    epsilons = all_parameters[:, -2]
    
    for j in range(num_stokeslets):
        # Scale the universal base force by this specific Stokeslet's coefficient
        base_force = all_parameters[j, 2:4]
        PCA_mode = int(all_parameters[j, -1])
        multiplier_array = coeff_time[:, PCA_mode]
        
        # Calculate the individual contribution
        u_j = compute_flow_velocity(
            target_pos=target_pos, 
            regStokes_pos=stokeslet_positions[j], 
            current_t=current_t, 
            base_force=base_force, 
            multiplier_array=multiplier_array, 
            window_map=window_map, 
            epsilon=epsilons[j], 
            dt=dt, 
            nu=nu, 
            rho=rho
        )
        
        # Superimpose (vector addition)
        total_u += u_j
        
    return total_u
    
# ---------------------------------------------------------
# PARALLEL GRID SOLVER
# ---------------------------------------------------------
# The parallel=True flag tells Numba to multi-thread the prange loops
@njit(parallel=True)
def compute_flow_field_grid(target_grid, 
                            all_parameters, 
                            current_t, 
                            coeff_time, 
                            window_map, 
                            dt=5e-3,
                            nu=168, 
                            rho=1/168
                           ) -> np.ndarray:
    """
    Evaluates the total flow field across an entire grid of points simultaneously 
    using multi-core CPU parallelization.

    Parameters
    ----------
    target_grid -> np.ndarray : A 3D array of shape (Nx, Ny, 2) representing the coordinates
                                of each point in the 2D space.
    all_parameters -> np.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    current_t -> float : current timepoint
    coeff_time -> np.ndarray : Scalar multipliers for force over one flagellar beat period
    window_map -> np.ndarray : time window beyond which the convolution integral will be truncated
    dt -> float : timestep
    nu -> int : kinematic viscosity
    rho -> float : density

    Returns
    -------
    np.ndarray : Fluid flow velocity at all grid points due to all the Stokeslets of a sperm cell
    """
    Nx, Ny, _ = target_grid.shape
    
    # Initialize an empty array of the exact same shape for the output velocities
    velocity_grid = np.zeros_like(target_grid)
    
    # prange distributes the outermost loop across your CPU cores
    for i in prange(Nx):
        for j in range(Ny):
                
            # Extract the 2D coordinate for this specific grid point
            target_pos = target_grid[i, j]
            
            # Compute the field using our superimposed function
            velocity_grid[i, j] = compute_total_velocity(
                target_pos=target_pos,
                all_parameters=all_parameters, 
                current_t=current_t,  
                coeff_time=coeff_time,
                window_map=window_map,
                dt=dt,
                nu=nu,
                rho=rho
            )
                
    return velocity_grid
    
@njit(parallel=True)
def compute_flow_field_over_time(target_grid, 
                                 all_parameters, 
                                 time_array, 
                                 coeff_time, 
                                 window_map, 
                                 dt=5e-3,
                                 nu=168, 
                                 rho=1/168):
    """
    Evaluates the total flow field across an entire grid over multiple time steps.

    Parameters
    ----------
    target_grid -> np.ndarray : A 3D array of shape (Nx, Ny, 2) representing the coordinates
                                of each point in the 2D space.
    all_parameters -> np.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    time_array -> np.ndarray : all the timepoints where the flow field is evaluated
    coeff_time -> np.ndarray : Scalar multipliers for force over one flagellar beat period
    window_map -> np.ndarray : time window beyond which the convolution integral will be truncated
    dt -> float : timestep
    nu -> int : kinematic viscosity
    rho -> float : density

    Returns
    -------
    np.ndarray : Fluid flow velocity at all grid points due to all the Stokeslets of a sperm cell at all timepoints.
    """
    Nt = len(time_array)
    Nx, Ny, _ = target_grid.shape
    
    # Initialize the gigantic array! 
    # Shape: (Number of time points, X points, Y points, 2 velocity components)
    velocity_grids = np.zeros((Nt, Nx, Ny, 2))
    
    # prange distributes the time steps across your CPU cores
    for t_idx in prange(Nt):
        current_t = time_array[t_idx]
        
        # Standard sequential loops for the spatial grid
        for i in range(Nx):
            for j in range(Ny):
                    
                target_pos = target_grid[i, j]
                
                # Compute the field and store it in the 4D array
                velocity_grids[t_idx, i, j] = compute_total_velocity(
                    target_pos=target_pos,
                    all_parameters=all_parameters, 
                    current_t=current_t,  
                    coeff_time=coeff_time,
                    window_map=window_map, 
                    dt=dt,
                    nu=nu,
                    rho=rho
                )
                
    return velocity_grids
    
# Here onwards is for steady regularised Stokeslets
@njit
def steady_gaussian_stokeslet(dx, dy, 
                              fx, fy,
                              coef,
                              eps_g,
                              mu=1
                             ):
    """
    Computes the steady flow field u, v induced by a regularized Stokeslet 
    with a Gaussian blob at a distance vector (dx, dy).

    Parameters
    ----------
    dx, dy -> floats : displacement vector from regularised Stokeslet to point of interest.
    fx, fy -> floats : force vector of the regularised Stokeslet
    coef -> float : time-dependent coefficient 
    eps_g -> float : radius of regularised Stokeslet
    mu -> float : dynamics viscosity, non-dimensionalised to 1

    Returns
    u, v : fluid velocity at the point of interest
    """
    r2 = dx**2 + dy**2
    r = math.sqrt(r2)
    
    # Numerical guard: Prevent division by zero if evaluating exactly AT the Stokeslet
    if r < 1e-12:
        multiplier = 1 / (3 * math.pi * math.sqrt(2*math.pi) * mu * eps_g)
        return coef * multiplier * fx, coef * multiplier * fy
        
    # Pre-compute repeated variables to save CPU cycles
    sqrt_2_eps = math.sqrt(2.0) * eps_g
    R = r / sqrt_2_eps
    
    erf_R = math.erf(R)
    exp_R2 = math.exp(-R**2)
    
    # --- Compute Bracket A (The delta_ij term) ---
    A_part1 = erf_R
    A_part2 = exp_R2 / (math.sqrt(math.pi) * R)
    A_part3 = (eps_g / r)**2 * erf_R
    A = A_part1 - A_part2 + A_part3
    
    # --- Compute Bracket B (The x_i x_j / r^2 term) ---
    B_part1 = erf_R
    B_part2 = 3 * exp_R2 / (math.sqrt(math.pi) * R)
    B_part3 = 3.0 * (eps_g / r)**2 * erf_R
    B = B_part1 + B_part2 - B_part3
    
    # --- Combine into final velocity vectors ---
    # Leading multiplier
    multiplier = 1.0 / (8.0 * math.pi * mu * r)
    
    # The dot product of Force and Distance (f_j x_j)
    f_dot_r = fx * dx + fy * dy
    
    # Final u_i calculations
    u = coef * multiplier * (A * fx + B * f_dot_r * dx / r2)
    v = coef * multiplier * (A * fy + B * f_dot_r * dy / r2)
    
    return u, v
    
# ---------------------------------------------------------
# NEW: MULTI-STOKESLET SUPERPOSITION
# ---------------------------------------------------------
@njit
def compute_total_velocity_steady(target_pos, 
                           all_parameters, 
                           current_t,  
                           coeff_time, 
                           mu=1
                          ) -> np.ndarray:
    """
    Calculates the combined flow field from all regularised Stokeslets of the same cell 
    using the principle of linear superposition.

    Parameters
    ----------
    target_pos -> np.ndarray : (x, y) coordinates of the target position
    all_parameters -> np.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    current_t -> float : current timepoint
    coeff_time -> np.ndarray : a 2D array recording the time-dependent coefficients of each regularised Stokeslet
    mu -> 1 : dynamics viscosity

    Returns
    -------
    total_u, total_v -> floats : Fluid flow velocity at the position of interest due to all the Stokeslets of a sperm cell
    """
    total_u = 0.0
    total_v = 0.0
    num_stokeslets = all_parameters.shape[0]
    stokeslet_positions = all_parameters[:, :2]
    epsilons = all_parameters[:, -2]
    
    time_index = int(round((current_t % 1.0) / 5e-3))
    # Failsafe for floating point rounding pushing index out of bounds
    if time_index >= coeff_time.shape[0]:
        time_index = 0
    
    for j in range(num_stokeslets):
        # Scale the universal base force by this specific Stokeslet's coefficient
        fx, fy = all_parameters[j, 2:4]
        PCA_mode = int(all_parameters[j, -1])
        dx = target_pos[0] - stokeslet_positions[j, 0]
        dy = target_pos[1] - stokeslet_positions[j, 1]
        current_coef = coeff_time[time_index, PCA_mode]
        
        # Calculate the individual contribution
        u_j, v_j = steady_gaussian_stokeslet(
            dx, dy,  
            fx, fy,  
            current_coef, # Pass the extracted coefficient
            eps_g=epsilons[j],
            mu=mu
        )
        
        # Superimpose (vector addition)
        total_u += u_j
        total_v += v_j
        
    return total_u, total_v
    
@njit(parallel=True)
def compute_flow_field_over_time_steady(target_grid, 
                                 all_parameters, 
                                 time_array, 
                                 coeff_time, 
                                 mu=1):
    """
    Evaluates the total flow field across an entire grid over multiple time steps.

    Parameters
    ----------
    target_grid -> np.ndarray : A 3D array of shape (Nx, Ny, 2) representing the coordinates
                                of each point in the 2D space.
    all_parameters -> np.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    time_array -> np.ndarray : all the timepoints where the flow field is evaluated
    coeff_time -> np.ndarray : Scalar multipliers for force over one flagellar beat period
    mu -> float : dynamics viscosity

    Returns
    -------
    np.ndarray : Fluid flow velocity at all grid points due to all the Stokeslets of a sperm cell at all timepoints.
    """
    Nt = len(time_array)
    Nx, Ny, _ = target_grid.shape
    
    # Initialize the gigantic array! 
    # Shape: (Number of time points, X points, Y points, 2 velocity components)
    velocity_grids = np.zeros((Nt, Nx, Ny, 2))
    
    # prange distributes the time steps across your CPU cores
    for t_idx in prange(Nt):
        current_t = time_array[t_idx]
        
        # Standard sequential loops for the spatial grid
        for i in range(Nx):
            for j in range(Ny):
                    
                target_pos = target_grid[i, j]
                
                # Compute the field and store it in the 4D array
                u, v = compute_total_velocity_steady(
                    target_pos=target_pos,
                    all_parameters=all_parameters, 
                    current_t=current_t,  
                    coeff_time=coeff_time,
                    mu=mu
                )

                velocity_grids[t_idx, i, j, 0] = u
                velocity_grids[t_idx, i, j, 1] = v
                
    return velocity_grids