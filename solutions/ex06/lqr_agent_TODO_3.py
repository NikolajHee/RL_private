        k = int(t / self.env.dt) 
        u = self.L[k] @ x + self.l[k] 