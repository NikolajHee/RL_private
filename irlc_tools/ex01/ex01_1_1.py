from typing import NamedTuple
import numpy as np

K = 4
ACTIONS = {(1, 0), (-1, 0), (0, 1), (0, -1)}
NX = 5
NY = 5
GOAL = (NX-1, NY-1)
J = np.zeros([NX, NY])


class GridworldState(NamedTuple):
    x: int
    y: int


def clip(val, lower, upper):
    return max(lower, min(upper, val))


def step(state: GridworldState, action):
    assert action in ACTIONS

    if (state.x, state.y) == GOAL:
        cost = 0.
        return state, cost
    cost = 1.
    x = clip(state.x + action[0], 0, NX-1)
    y = clip(state.y + action[1], 0, NY-1)

    return GridworldState(x, y), cost


for k in range(K):
    V = J.copy()
    for x in range(NX):
        for y in range(NY):
            best = np.inf
            for a in ACTIONS:
                s, g = step(GridworldState(x, y), a)
                val = g + J[s.x, s.y]
                if val < best:
                    best = val
                    V[x, y] = val
    J = V
    print(J)



