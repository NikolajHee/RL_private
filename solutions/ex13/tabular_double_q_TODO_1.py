        return self.random_pi(s) if np.random.rand() < self.epsilon else a1[np.argmax(Q + np.random.rand(len(Q)) * 1e-8)] 