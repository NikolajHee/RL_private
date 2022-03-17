            X = np.asarray(self.buffer.x) 
            Y = np.asarray(self.buffer.xp)
            U = np.asarray(self.buffer.u)
            cost = self.env.discrete_model.cost
            A, B, C = solve_linear_problem_simple(Y=Y.T, X=X.T, U=U.T, lamb=self.lamb)
            (self.L, self.l), (V, v, vc) = LQR(A=[A] * N, B=[B] * N, d=[C]*N, Q=[cost.Q] * N, R=[cost.R] * N, q=[cost.q] * N, qc=[cost.qc] * N) 