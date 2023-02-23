        t = info['time_seconds']
        if t > self.ts_grid[-1]:
            print("Simulation time is", t, "which exceeds the maximal planning horizon t_F =", self.ts_grid[-1])
            raise Exception("Time exceed agents planning horizon")

        u = self.ufun(t)
        u = np.asarray(self.env.discrete_model.continious_actions2discrete_actions(u)) 