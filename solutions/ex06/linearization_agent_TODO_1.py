        xp, A, B = model.f(xbar, ubar, k=0, compute_jacobian=True) 
        d = xp - A @ xbar - B @ ubar 