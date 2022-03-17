            k = int(t / self.dt)  # current timepoint 
            u = self.L[k] @ x + self.l[k] 