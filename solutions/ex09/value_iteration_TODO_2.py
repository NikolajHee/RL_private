        Q = {a: v-(1e-8*a if isinstance(a, int) else 0) for a,v in qs_(mdp, s, gamma, V).items()} 
        pi[s] = max(Q, key=Q.get) 