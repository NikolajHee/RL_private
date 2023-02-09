    for k in range(model.N): 
        u = pi(x, k) # Generate the action u = ... here using the policy
        w = model.w_rnd(x, u, k) # This is required; just pass them to the transition function
        J += model.g(x, u, w, k) # Add cost term g_k to the cost of the episode
        x = model.f(x, u, w, k) # Update J and generate the next value of x.
        trajectory.append(x) # update the trajectory
    J += model.gN(x) # Add last cost term env.gN(x) to J.  