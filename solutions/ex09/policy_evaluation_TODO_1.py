    return {a: sum([p*(r+ (gamma*v[sp] if not mdp.is_terminal(sp) else 0)) for (sp,r), p in mdp.Psr(s,a).items()]) for a in mdp.A(s)}