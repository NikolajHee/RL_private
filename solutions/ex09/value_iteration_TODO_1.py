            v, V[s] = V[s], max(qs_(mdp, s, gamma, V).values()) if len(mdp.A(s))>0 else 0    
            Delta = max(Delta, np.abs(v - V[s])) 