        w = np.random.choice(3, p=(.1, .7, .2))  # Generate random disturbance
        s_next = max(0, min(2, self.s-w+a))      # next state; x_{k+1} =  f_k(x_k, u_k, w_k) 