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
        
    # Calculate the physical start time
    t_start = max(0.0, current_t - window)
    
    # Convert absolute times into global integer indices
    # (round() protects against floating-point precision errors like 1.999999 / 0.1)
    idx_start = int(round(t_start / dt))
    idx_end = int(round(current_t / dt))
        
    N_period = len(multiplier_array)
    integral_sum = np.zeros(2, dtype=float)
    
    # Traverse exact indices in the integration window
    for i in range(idx_start, idx_end):
        # Physical absolute times for Stokeslet evaluation
        tau1 = i * dt
        tau2 = (i + 1) * dt
        
        # Wrapped indices for the single-period multiplier array
        wrap_idx1 = i % N_period
        wrap_idx2 = (i + 1) % N_period
        
        # Evaluate at tau1
        dt1 = current_t - tau1
        R1 = calc_stokeslet_tensor(r_vec, epsilon, dt1, nu)
        F1 = base_force * multiplier_array[wrap_idx1]
        integrand1 = R1 @ F1
        
        # Evaluate at tau2
        dt2 = current_t - tau2
        R2 = calc_stokeslet_tensor(r_vec, epsilon, dt2, nu)
        F2 = base_force * multiplier_array[wrap_idx2]
        integrand2 = R2 @ F2
        
        # Add to sum (Trapezoidal rule with constant dt)
        integral_sum += 0.5 * (integrand1 + integrand2) * dt
        
    prefactor = 1.0 / (4 * math.pi * rho)
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