        u = -self.pid.pi(x[-2] + (.12*x[0]+.02*x[1] if self.balance_to_x0 else 0) )
        u = np.clip(u, -self.env.max_force, self.env.max_force) # Clip max torque. 