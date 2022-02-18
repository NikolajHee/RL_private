                if xj not in J[k + 1] or c + J[k][xi] < J[k + 1][xj]:
                    J[k + 1][xj] = c + J[k][xi]
                    pi[k][xj] = (a, xi) 