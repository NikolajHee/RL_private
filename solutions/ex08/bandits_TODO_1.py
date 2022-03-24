        reward = self.q_star[a] + np.random.randn() 
        regret = self.q_star[self.optimal_action] - self.q_star[a] 