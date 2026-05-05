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
    
    zeta2 = zeta**2
    zeta3 = zeta**3
    sqrt_pi = jnp.sqrt(jnp.pi)
    
    exp_term = jnp.exp(-(r**2) / (4.0 * zeta2))
    erf_term = jsp.erf(r / (2.0 * zeta))
    
    is_close = r < 1e-12
    safe_r = jnp.where(is_close, 1.0, r)
    
    r2 = safe_r**2
    r3 = safe_r**3
    
    A = (1.0 / (2.0 * sqrt_pi * zeta3)) * exp_term \
      - (1.0 / r3) * erf_term \
      + (1.0 / (sqrt_pi * zeta * r2)) * exp_term
      
    B = (3.0 / r3) * erf_term \
      - (3.0 / (sqrt_pi * zeta * r2)) * exp_term \
      - (1.0 / (2.0 * sqrt_pi * zeta3)) * exp_term
      
    safe_r_hat = r_vec / safe_r
    rr_tensor = jnp.outer(safe_r_hat, safe_r_hat)
    
    tensor_standard = jnp.eye(2) * A + rr_tensor * B
    
    limit_val = 1.0 / (3.0 * sqrt_pi * zeta3)
    tensor_limit = jnp.eye(2) * limit_val

    return jnp.where(is_close, tensor_limit, tensor_standard)
    
# ---------------------------------------------------------
# 2. CONVOLUTION INTEGRATION (Moving Source)
# ---------------------------------------------------------
@jax.jit
def compute_flow_velocity(target_pos, 
                          regStokes_local,    
                          base_force_local,   
                          sperm_pos_history,  
                          sperm_theta_history,
                          current_t, 
                          multiplier_array,
                          epsilon, 
                          dt_int,            
                          dt_force=5e-3, 
                          nu=168.0, 
                          rho=1.0/168.0):
    
    """
    Computes the flow velocity vector u at target_pos due to a moving regularised Stokeslet.

    Parameters
    ----------
    target_pos -> jnp.ndarray : (x, y) coordinates of the target position
    regStokes_local -> jnp.ndarray : (x, y) Origin of regularised Stokeslet in local body frame
    base_force_local -> jnp.ndarray : base force of the regularised Stokeslet in local body frame
    sperm_pos_history -> jnp.ndarray : coordinates of the sperm cell from t=0 to the current time, at time steps of dt_force
    sperm_theta_history -> jnp.ndarray : orientation of the sperm cell from t=0 to the current_time, at time steps of dt_force
    current_t -> float : current timepoint
    multiplier_array -> jnp.ndarray : 1D array of scalar multipliers for base_force over one flagellar beat period
    epsilon -> float : radius of the regularised Stokeslet
    dt_int -> float : timestep used for numerical integration
    dt_force -> float : timestep of the coefficient data
    nu -> float : kinematic viscosity
    rho -> float : density

    Returns
    -------
    jnp.ndarray : Fluid flow velocity at the position of interest
    """

    global_window = 4.41
    t_start = jnp.maximum(0.0, current_t - global_window)
    
    idx_start = jnp.round(t_start / dt_int).astype(jnp.int32)
    idx_end = jnp.round(current_t / dt_int).astype(jnp.int32)

    N_period = multiplier_array.shape[0]
    T_period = N_period * dt_force
    
    def get_force_magnitude(tau):
        """Interpolates the scalar force multiplier"""
        idx_float = (tau % T_period) / dt_force
        idx_low = jnp.floor(idx_float).astype(jnp.int32)
        idx_high = (idx_low + 1) % N_period
        w = idx_float - idx_low
        return multiplier_array[idx_low] * (1.0 - w) + multiplier_array[idx_high] * w

    def get_kinematics(tau):
        """Interpolates the sperm's global position and orientation at time tau"""
        
        idx_float = tau / dt_force
        
        idx_low = jnp.floor(idx_float).astype(jnp.int32)
        idx_high = idx_low + 1
        
        # THE FIX: Calculate the maximum valid index based on the CURRENT time
        current_max_idx = jnp.round(current_t / dt_force).astype(jnp.int32)
        
        # Clamp both indices so they can NEVER exceed the current known timestep
        idx_low = jnp.clip(idx_low, 0, current_max_idx)
        idx_high = jnp.clip(idx_high, 0, current_max_idx)
        
        # Interpolation weight
        w = idx_float - idx_low
        
        # Linearly interpolate between the two coarse history points
        pos = sperm_pos_history[idx_low] * (1.0 - w) + sperm_pos_history[idx_high] * w
        theta = sperm_theta_history[idx_low] * (1.0 - w) + sperm_theta_history[idx_high] * w
        
        return pos, theta

    def rotate_2d(vec, theta):
        """Rotates a local 2D vector into the global frame"""
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        return jnp.array([vec[0]*c - vec[1]*s, vec[0]*s + vec[1]*c])

    initial_integral = jnp.zeros(2, dtype=jnp.float32)

    def bulk_body_fn(i, integral_sum):
        """Executes one step of standard trapezoidal integration with dynamic kinematics"""
        tau1 = i * dt_int
        tau2 = (i + 1) * dt_int
        
        # --- KINEMATICS AT TAU 1 ---
        pos1, theta1 = get_kinematics(tau1)
        global_stokes_pos1 = pos1 + rotate_2d(regStokes_local, theta1)
        r_vec1 = target_pos - global_stokes_pos1
        
        F_mag1 = get_force_magnitude(tau1)
        F_global1 = rotate_2d(base_force_local * F_mag1, theta1)
        
        # --- KINEMATICS AT TAU 2 ---
        pos2, theta2 = get_kinematics(tau2)
        global_stokes_pos2 = pos2 + rotate_2d(regStokes_local, theta2)
        r_vec2 = target_pos - global_stokes_pos2
        
        F_mag2 = get_force_magnitude(tau2)
        F_global2 = rotate_2d(base_force_local * F_mag2, theta2)
        
        # --- EVALUATE TENSORS ---
        R1 = calc_stokeslet_tensor(r_vec1, epsilon, current_t - tau1, nu)
        R2 = calc_stokeslet_tensor(r_vec2, epsilon, current_t - tau2, nu)
        
        integrand1 = jnp.dot(R1, F_global1)
        integrand2 = jnp.dot(R2, F_global2)
        
        return integral_sum + 0.5 * (integrand1 + integrand2) * dt_int

    final_integral = jax.lax.fori_loop(idx_start, idx_end, bulk_body_fn, initial_integral)

    prefactor = 1.0 / (4.0 * jnp.pi * rho)
    return prefactor * final_integral
    
# ---------------------------------------------------------
# NEW: MULTI-STOKESLET SUPERPOSITION (JAX Version)
# ---------------------------------------------------------
@jax.jit
def compute_total_velocity(target_pos, 
                           all_parameters,     
                           sperm_pos_history,  
                           sperm_theta_history,
                           current_t,  
                           coeff_time,
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
    all_parameters -> jnp.ndarray : a 2D array recording the origin, radius, force, and PCA mode of each regularised Stokeslet, in body frame
    sperm_pos_history -> jnp.ndarray : coordinates of the sperm cell from t=0 to the current time, at time steps of dt_force
    sperm_theta_history -> jnp.ndarray : orientation of the sperm cell from t=0 to the current_time, at time steps of dt_force
    current_t -> float : current timepoint
    coeff_time -> jnp.ndarray : Scalar multipliers for force over one flagellar beat period
    dt_int -> float : timestep used for numerical integration
    dt -> float : timestep
    nu -> float : kinematic viscosity
    rho -> float : density

    Returns
    -------
    jnp.ndarray : Fluid flow velocity at the position of interest due to all the Stokeslets of a sperm cell
    """
    
    # 1. Slice out the individual local parameter arrays
    stokeslet_local_pos = all_parameters[:, :2]
    base_forces_local = all_parameters[:, 2:4]
    epsilons = all_parameters[:, -2]
    pca_modes = all_parameters[:, -1].astype(jnp.int32)
    
    # 2. Vectorized Multiplier Extraction
    multiplier_arrays = coeff_time[:, pca_modes].T
    
    # 3. Create the VMAP version
    # Note: History arrays get 'None' because all Stokeslets belong to the SAME cell 
    # and therefore share the exact same body trajectory.
    vmapped_compute = jax.vmap(
        compute_flow_velocity,
        in_axes=(
            None,  # target_pos
            0,     # regStokes_local (unique per stokeslet)
            0,     # base_force_local (unique per stokeslet)
            None,  # sperm_pos_history (shared!)
            None,  # sperm_theta_history (shared!)
            None,  # current_t
            0,     # multiplier_array (unique per stokeslet)
            0,     # epsilon (unique per stokeslet)
            None,  # dt_int
            None,  # dt_force
            None,  # nu
            None   # rho
        )
    )
    
    # 4. Fire the vectorized function
    all_velocities = vmapped_compute(
        target_pos, 
        stokeslet_local_pos, 
        base_forces_local, 
        sperm_pos_history,
        sperm_theta_history,
        current_t, 
        multiplier_arrays,
        epsilons, 
        dt_int,
        dt, 
        nu, 
        rho
    )
    
    # 5. Superimpose
    total_u = jnp.sum(all_velocities, axis=0)
    return total_u
    
# ---------------------------------------------------------
# NEW: PARALLEL GRID SOLVER (JAX Version)
# ---------------------------------------------------------
@jax.jit
def compute_flow_field_grid(target_grid, 
                            all_parameters, 
                            sperm_pos_history,
                            sperm_theta_history,
                            current_t, 
                            coeff_time, 
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
    sperm_pos_history -> jnp.ndarray : coordinates of the sperm cell from t=0 to the current time, at time steps of dt_force
    sperm_theta_history -> jnp.ndarray : orientation of the sperm cell from t=0 to the current_time, at time steps of dt_force
    current_t -> float : current timepoint
    coeff_time -> jnp.ndarray : Scalar multipliers for force over one flagellar beat period
    dt_int -> float : timestep used for numerical integration
    dt -> float : timestep
    nu -> int : kinematic viscosity
    rho -> float : density

    Returns
    -------
    jnp.ndarray : Fluid flow velocity at all grid points due to all the Stokeslets of a sperm cell
    """

    # 1. Map over Y-axis
    vmap_y = jax.vmap(
        compute_total_velocity,
        in_axes=(0, None, None, None, None, None, None, None, None, None)
    )
    
    # 2. Map over X-axis
    vmap_x_y = jax.vmap(
        vmap_y,
        in_axes=(0, None, None, None, None, None, None, None, None, None)
    )
    
    # 3. Execute
    velocity_grid = vmap_x_y(
        target_grid,
        all_parameters,
        sperm_pos_history,
        sperm_theta_history,
        current_t,
        coeff_time,
        dt_int,
        dt,
        nu,
        rho
    )
    
    return velocity_grid