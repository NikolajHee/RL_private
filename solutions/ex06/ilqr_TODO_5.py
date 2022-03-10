    l, L = [np.zeros((m,))] * N, [np.zeros((m, n))] * N  
    x_bar, u_bar = forward_pass(env, x_bar, u_bar, L=L, l=l)  