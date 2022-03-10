    fs = [(v[1],v[2]) for v in [model.f(x, u, i, compute_jacobian=True) for i, (x, u) in enumerate(zip(x_bar[:-1], u_bar))]]  
    f_x, f_u = zip(*fs) 