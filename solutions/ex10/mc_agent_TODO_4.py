        self.episode.append((s, a, r))
        if done:
            returns = get_MC_return_SA(self.episode, self.gamma, self.first_visit)
            for sa, G in returns:
                s,a = sa
                if self.alpha is None:
                    self.returns_sum[sa] += G
                    self.returns_count[sa] += 1
                    self.Q[s, a] = self.returns_sum[sa] / self.returns_count[sa]
                else:
                    self.Q[s, a] += self.alpha * (G - self.Q[s, a])
            self.episode = [] 