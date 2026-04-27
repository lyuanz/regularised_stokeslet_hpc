import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

# ---------------------------------------------------------
# 1. CORE MATH: STOKESLET TENSOR (JAX Version)
# ---------------------------------------------------------
@jax.jit
def calc_stokeslet_tensor(r_vec, 
                          epsilon,
                          delta_t, 
                          nu=168.0):
    
    """
    Computes the 2x2 unsteady regularised Stokeslet tensor R(r, t-tau).

    Parameters
    ----------
    r_vec -> jnp.ndarray : (x, y) displacement vector of the point of interest with respect to the centre of regularised Stokeslet
    epsilon -> float : radius of regularised Stokeslet
    delta_t -> float : Duration from force pulse application
    nu -> float : kinematic viscosity

    Returns
    -------
    R(r, t-tau) -> jnp.ndarray : unsteady regularised Stokeslet tensor at position r from the source, at time t-tau
    """
    
    r = jnp.linalg.norm(r_vec)
    zeta = jnp.sqrt(nu * delta_t + 0.5 * epsilon**2)
    
    # Precompute common terms
    zeta2 = zeta**2
    zeta3 = zeta**3
    sqrt_pi = jnp.sqrt(jnp.pi)
    
    exp_term = jnp.exp(-(r**2) / (4.0 * zeta2))
    erf_term = jsp.erf(r / (2.0 * zeta))
    
    # --- The JAX Division-by-Zero Fix ---
    # Create a safe_r that is 1.0 when r is near zero to prevent NaNs during compilation
    is_close = r < 1e-12
    safe_r = jnp.where(is_close, 1.0, r)
    
    r2 = safe_r**2
    r3 = safe_r**3
    
    # Term A (Coefficient for Identity Matrix)
    A = (1.0 / (2.0 * sqrt_pi * zeta3)) * exp_term \
      - (1.0 / r3) * erf_term \
      + (1.0 / (sqrt_pi * zeta * r2)) * exp_term
      
    # Term B (Coefficient for r_hat outer product r_hat)
    B = (3.0 / r3) * erf_term \
      - (3.0 / (sqrt_pi * zeta * r2)) * exp_term \
      - (1.0 / (2.0 * sqrt_pi * zeta3)) * exp_term
      
    # --- Vectorized Outer Product ---
    # JAX replaces the slow nested for-loops with a single functional command
    safe_r_hat = r_vec / safe_r
    rr_tensor = jnp.outer(safe_r_hat, safe_r_hat)
    
    # Assemble the standard 2x2 tensor
    tensor_standard = jnp.eye(2) * A + rr_tensor * B
    
    # Calculate the limit tensor for r -> 0
    limit_val = 1.0 / (3.0 * sqrt_pi * zeta3)
    tensor_limit = jnp.eye(2) * limit_val

    # JAX conditional: return tensor_limit if r is tiny, otherwise return tensor_standard
    return jnp.where(is_close, tensor_limit, tensor_standard)
    
# ---------------------------------------------------------
# 2. TIME WINDOW LOGIC (JAX Version)
# ---------------------------------------------------------
@jax.jit
def get_time_window(r, 
                    window_map):
    """
    Looks up the time window for a given distance r.

    Parameters
    ----------
    r -> float : distance between position of interest and the source
    window_map -> jnp.ndarray : An array recording the range of r values and the time window of each range, i.e. [[r_min, r_max, time_window], [...]]

    Returns
    -------
    time_window -> float : The time window
    """
    # 1. Create a boolean array checking the condition for every row simultaneously
    # is_in_range will look like [False, False, True, False, ...]
    is_in_range = (window_map[:, 0] <= r) & (r < window_map[:, 1])
    
    # 2. Check if ANY row actually matched
    has_match = jnp.any(is_in_range)
    
    # 3. Find the index of the True value. 
    # (If all are False, argmax returns 0, which we catch in the next step)
    match_idx = jnp.argmax(is_in_range)
    
    # 4. JAX Conditional: If we found a match, use match_idx. 
    # Otherwise, use the last index (shape[0] - 1) as the fallback.
    last_idx = window_map.shape[0] - 1
    final_idx = jnp.where(has_match, match_idx, last_idx)
    
    # 5. Extract and return the time_window value
    return window_map[final_idx, 2]
    
# ---------------------------------------------------------
# 3. CONVOLUTION INTEGRATION (JAX Version - Uniform Grid)
# ---------------------------------------------------------
@jax.jit
def compute_flow_velocity(target_pos, 
                          regStokes_pos, 
                          current_t, 
                          base_force, 
                          multiplier_array,  
                          window_map, 
                          epsilon, 
                          dt_int,    # The step size for the numerical integration
                          dt_force=5e-3,  # The spacing of your coefficient data
                          nu=168.0, 
                          rho=1.0/168.0):
    """
    Computes the flow velocity vector u at target_pos due to a regularised Stokeslet.
    Performs uniform trapezoidal numerical integration, smoothly interpolating force data.

    Parameters
    ----------
    target_pos -> jnp.ndarray : (x, y) coordinates of the target position
    regStokes_pos -> jnp.ndarray : (x, y) coordinates of the origin of regularised Stokeslet
    current_t -> float : current timepoint
    base_force -> jnp.ndarray : base force of the regularised Stokeslet
    multiplier_array -> jnp.ndarray : 1D array of scalar multipliers for base_force over one flagellar beat period
    window_map -> float : time window beyond which the convolution integral will be truncated
    epsilon -> float : radius of the regularised Stokeslet
    dt_int -> float : timestep used for numerical integration
    dt_force -> float : timestep of the coefficient data
    nu -> float : kinematic viscosity
    rho -> float : density

    Returns
    -------
    jnp.ndarray : Fluid flow velocity at the position of interest
    """
    r_vec = target_pos - regStokes_pos
    r = jnp.linalg.norm(r_vec)
    
    window = get_time_window(r, window_map)
    t_start = jnp.maximum(0.0, current_t - window)
    
    # Calculate loop bounds based on the INTEGRATION step size
    idx_start = jnp.round(t_start / dt_int).astype(jnp.int32)
    idx_end = jnp.round(current_t / dt_int).astype(jnp.int32)
        
    N_period = multiplier_array.shape[0]
    T_period = N_period * dt_force
    
    # ---------------------------------------------------------
    # UNIFIED FORCE INTERPOLATOR
    # ---------------------------------------------------------
    def get_force(tau):
        """Linearly interpolates force at any time tau using the base dt_force"""
        idx_float = (tau % T_period) / dt_force
        idx_low = jnp.floor(idx_float).astype(jnp.int32)
        idx_high = (idx_low + 1) % N_period
        w = idx_float - idx_low
        return base_force * (multiplier_array[idx_low] * (1.0 - w) + multiplier_array[idx_high] * w)

    # Initialize our integral state
    initial_integral = jnp.zeros(2, dtype=jnp.float32)

    # ---------------------------------------------------------
    # STANDARD BULK INTEGRATION (JAX fori_loop)
    # ---------------------------------------------------------
    def bulk_body_fn(i, integral_sum):
        """Executes one step of standard trapezoidal integration using dt_int"""
        tau1 = i * dt_int
        tau2 = (i + 1) * dt_int
        
        F1 = get_force(tau1)
        F2 = get_force(tau2)
        
        R1 = calc_stokeslet_tensor(r_vec, epsilon, current_t - tau1, nu)
        R2 = calc_stokeslet_tensor(r_vec, epsilon, current_t - tau2, nu)
        
        integrand1 = jnp.dot(R1, F1)
        integrand2 = jnp.dot(R2, F2)
        
        return integral_sum + 0.5 * (integrand1 + integrand2) * dt_int

    # Run the uniform loop
    final_integral = jax.lax.fori_loop(idx_start, idx_end, bulk_body_fn, initial_integral)

    prefactor = 1.0 / (4.0 * jnp.pi * rho)
    return prefactor * final_integral
    
# ---------------------------------------------------------
# NEW: MULTI-STOKESLET SUPERPOSITION (JAX Version)
# ---------------------------------------------------------
@jax.jit
def compute_total_velocity(target_pos, 
                           all_parameters, 
                           current_t,  
                           coeff_time,
                           window_map,
                           dt_int, 
                           dt=5e-3,
                           nu=168.0, 
                           rho=1.0/168.0):
    """
    Calculates the combined flow field from all regularised Stokeslets of the same cell 
    using the principle of linear superposition.

    Parameters
    ----------
    target_pos -> jnp.ndarray : (x, y) coordinates of the target position
    all_parameters -> jnp.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    current_t -> float : current timepoint
    coeff_time -> jnp.ndarray : Scalar multipliers for force over one flagellar beat period
    window_map -> jnp.ndarray : time window beyond which the convolution integral will be truncated
    dt_int -> float : timestep used for numerical integration
    dt -> float : timestep
    nu -> float : kinematic viscosity
    rho -> float : density

    Returns
    -------
    jnp.ndarray : Fluid flow velocity at the position of interest due to all the Stokeslets of a sperm cell
    """
    # 1. Slice out the individual parameter arrays for all Stokeslets at once
    stokeslet_positions = all_parameters[:, :2]
    base_forces = all_parameters[:, 2:4]
    epsilons = all_parameters[:, -2]
    
    # Cast PCA modes to integers so they can be used as indices
    pca_modes = all_parameters[:, -1].astype(jnp.int32)
    
    # 2. Vectorized Multiplier Extraction:
    # coeff_time is shape (time_steps, num_modes)
    # By indexing it with our pca_modes array, we get shape (time_steps, num_stokeslets)
    # We transpose (.T) it to get (num_stokeslets, time_steps) to match our vmap axis
    multiplier_arrays = coeff_time[:, pca_modes].T
    
    # 3. Create the massively parallel VMAP version of your compute function
    # in_axes tells JAX how to handle each argument:
    # None = Broadcast the exact same value to every Stokeslet calculation
    # 0 = Slice this array along axis 0, giving each Stokeslet its own specific row
    vmapped_compute = jax.vmap(
        compute_flow_velocity,
        in_axes=(
            None,  # target_pos: same for all
            0,     # regStokes_pos: unique per stokeslet
            None,  # current_t: same for all
            0,     # base_force: unique per stokeslet
            0,     # multiplier_array: unique per stokeslet
            None,  # window_map: same for all
            0,     # epsilon: unique per stokeslet
            None,  # dt_int: same
            None,  # dt: same
            None,  # nu: same
            None   # rho: same
        )
    )
    
    # 4. Fire the vectorized function. 
    # The GPU will calculate all Stokeslets simultaneously!
    # Returns an array of shape (num_stokeslets, 2)
    all_velocities = vmapped_compute(
        target_pos, 
        stokeslet_positions, 
        current_t, 
        base_forces, 
        multiplier_arrays, 
        window_map, 
        epsilons,
        dt_int, 
        dt, 
        nu, 
        rho
    )
    
    # 5. Superimpose (Vector addition by summing along the Stokeslet axis)
    total_u = jnp.sum(all_velocities, axis=0)
    
    return total_u
    
# ---------------------------------------------------------
# NEW: PARALLEL GRID SOLVER (JAX Version)
# ---------------------------------------------------------
@jax.jit
def compute_flow_field_grid(target_grid, 
                            all_parameters, 
                            current_t, 
                            coeff_time, 
                            window_map,
                            dt_int, 
                            dt=5e-3,
                            nu=168.0, 
                            rho=1.0/168.0):
    """
    Evaluates the total flow field across an entire grid of points simultaneously.

    Parameters
    ----------
    target_grid -> jnp.ndarray : A 3D array of shape (Nx, Ny, 2) representing the coordinates
                                of each point in the 2D space.
    all_parameters -> jnp.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    current_t -> float : current timepoint
    coeff_time -> jnp.ndarray : Scalar multipliers for force over one flagellar beat period
    window_map -> jnp.ndarray : time window beyond which the convolution integral will be truncated
    dt_int -> float : timestep used for numerical integration
    dt -> float : timestep
    nu -> int : kinematic viscosity
    rho -> float : density

    Returns
    -------
    jnp.ndarray : Fluid flow velocity at all grid points due to all the Stokeslets of a sperm cell
    """
    # ---------------------------------------------------------
    # BUILD THE VMAP STACK
    # ---------------------------------------------------------
    
    # 1. Map over the Y-axis (Inner loop replacement)
    # Base function expects: target_pos of shape (2,)
    # We slice target_pos (axis 0), everything else gets None (broadcast)
    vmap_y = jax.vmap(
        compute_total_velocity,
        in_axes=(0, None, None, None, None, None, None, None, None)
    )
    
    # 2. Map over the X-axis (Outer loop replacement)
    # We vmap our already-vmapped function.
    # It now expects target_grid of shape (Nx, Ny, 2)
    vmap_x_y = jax.vmap(
        vmap_y,
        in_axes=(0, None, None, None, None, None, None, None, None)
    )
    
    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------
    # Fire the calculation! 
    # JAX natively returns the array in the shape it was fed: (Nx, Ny, 2)
    velocity_grid = vmap_x_y(
        target_grid,
        all_parameters,
        current_t,
        coeff_time,
        window_map,
        dt_int,
        dt,
        nu,
        rho
    )
    
    return velocity_grid
    
# ---------------------------------------------------------
# NEW: TIME-RESOLVED PARALLEL GRID SOLVER (JAX Version)
# ---------------------------------------------------------
@jax.jit
def compute_flow_field_over_time(target_grid, 
                                 all_parameters, 
                                 time_array, 
                                 coeff_time, 
                                 window_map,
                                 dt_int,
                                 dt=5e-3,
                                 nu=168.0, 
                                 rho=1.0/168.0):
    """
    Evaluates the total flow field across an entire grid over multiple time steps.

    Parameters
    ----------
    target_grid -> jnp.ndarray : A 3D array of shape (Nx, Ny, 2) representing the coordinates
                                of each point in the 2D space.
    all_parameters -> jnp.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet
    time_array -> jnp.ndarray : all the timepoints where the flow field is evaluated
    coeff_time -> jnp.ndarray : Scalar multipliers for force over one flagellar beat period
    window_map -> jnp.ndarray : time window beyond which the convolution integral will be truncated
    dt_int -> float : timestep used for numerical integration
    dt -> float : timestep
    nu -> int : kinematic viscosity
    rho -> float : density

    Returns
    -------
    jnp.ndarray : Fluid flow velocity at all grid points due to all the Stokeslets of a sperm cell at all timepoints.
    """
    # ---------------------------------------------------------
    # BUILD THE VMAP STACK
    # ---------------------------------------------------------
    # Base function signature:
    # compute_total_velocity(target_pos, all_parameters, current_t, coeff_time, window_map, dt, nu, rho)
    
    # 1. Map over the Y-axis of the grid
    # We tell JAX to slice the 0th argument (target_pos) along its rows.
    # The function now accepts a 2D array of shape (Ny, 2) instead of a single (2,) point.
    vmap_y = jax.vmap(
        compute_total_velocity, 
        in_axes=(0, None, None, None, None, None, None, None, None)
    )
    
    # 2. Map over the X-axis of the grid
    # We vmap our already-vmapped function! We slice the 0th argument again.
    # The function now accepts a 3D array of shape (Nx, Ny, 2).
    vmap_x_y = jax.vmap(
        vmap_y, 
        in_axes=(0, None, None, None, None, None, None, None, None)
    )
    
    # 3. Map over Time
    # We vmap one last time. This time, we do NOT slice target_grid (it gets passed as 'None').
    # Instead, we slice the 2nd argument (time_array).
    # The function now evaluates the whole spatial grid across the time array of shape (Nt,).
    vmap_t_x_y = jax.vmap(
        vmap_x_y,
        in_axes=(None, None, 0, None, None, None, None, None, None)
    )
    
    # ---------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------
    # Fire the massively parallel, fully-vectorized function.
    # The output natively constructs itself into the requested shape: (Nt, Nx, Ny, 2)
    velocity_grids = vmap_t_x_y(
        target_grid, 
        all_parameters, 
        time_array, 
        coeff_time, 
        window_map, 
        dt_int, 
        dt, 
        nu, 
        rho
    )
    
    return velocity_grids