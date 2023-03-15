        z, z0, z_lb, z_ub = z + xs[k], z0 + list(guess['x'](tk).flat), z_lb + x_low, z_ub + x_high 
        z, z0, z_lb, z_ub = z + us[k], z0 + list(guess['u'](tk).flat), z_lb + u_low, z_ub + u_high 