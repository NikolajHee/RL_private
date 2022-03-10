        (f_x, f_u), (L, L_x, L_u, L_xx, L_ux, L_uu) = get_derivatives(env, x_bar, u_bar)  
        J_bar = sum(L)  