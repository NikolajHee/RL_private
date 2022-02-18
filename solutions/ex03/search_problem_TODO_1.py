        if k == self.env.N: 
            return {0: ( self.terminal_state, self.env.gN(s))}
        return {u: ((self.env.f(s, u, None, k), k+1), self.env.g(s, u, None, k)) for u in self.env.A(s, k)} 