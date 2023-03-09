            X,U,XP = self.buffer.get_data() 
            N = self.horizon_length
            cost = self.env.discrete_model.cost
            A, B, C = solve_linear_problem_simple(Y=XP, X=X, U=U, lamb=self.lamb)
            (self.L, self.l), (V, v, vc) = LQR(A=[A] * N, B=[B] * N, d=[C]*N, Q=[cost.Q] * N, R=[cost.R] * N, q=[cost.q]*N, qc=[cost.qc]*N) 