        for a in state.A(): 
            sp = state.f(a)
            transitions[a] = (sp, s0-sp.getScore()) 