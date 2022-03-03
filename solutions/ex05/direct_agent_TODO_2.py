        if t > self.ts_grid[-1]:
            print("Bad time!", t, self.ts_grid[-1])
            raise Exception("Time exceed agents planning horizon")

        u = self.ufun(t)
        u = np.asarray(self.env.discrete_model.continious_actions2discrete_actions(u)) 