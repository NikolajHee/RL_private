    simple_bounds.update({'x': Bounds([-2 * dist, -np.inf, -2 * np.pi, -np.inf], [2 * dist, np.inf, 2 * np.pi, np.inf]), 
                         'u': Bounds([-maxForce], [maxForce]),
                         'x0': Bounds([0, 0, np.pi, 0], [0, 0, np.pi, 0]),
                         'xF': Bounds([dist, 0, 0, 0], [dist, 0, 0, 0])}) 