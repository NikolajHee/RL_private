        self.Q.w += self.alpha * delta * self.Q.x(s,a)  # Update q(s,a)/weights given change in q-values: delta = [G-\hat{q}(..)]