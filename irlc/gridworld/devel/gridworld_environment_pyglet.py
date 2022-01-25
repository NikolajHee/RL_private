# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.gridworld.gridworld_environments import GridworldEnvironment, grid_book_grid
import numpy as np
from pyglet.window import key
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from gym.spaces.discrete import Discrete
from irlc.ex09.mdp import MDP2GymEnv
from irlc.gridworld.gridworld_mdp import GridworldMDP, FrozenGridMDP
from irlc.gridworld import gridworld_graphics_display
from irlc import Timer
from irlc.gridworld.devel.gridworld_graphics import GridworldViewer

class GridworldPyglet(GridworldEnvironment):
    def render(self, mode='human', state=None, agent=None, v=None, Q=None, pi=None, policy=None, v2Q=None, gamma=0,
               method_label="", label=None):

        gridSize = int(150 * self.zoom)

        if self.viewer is None:
            self.viewer = GridworldViewer(self.mdp, gridSize)

        if label is None:
            label = f"{method_label} AFTER {self.render_steps} STEPS"

        if state is None:
            state = self.state

        avail_modes = []
        if agent != None:
            label = (agent.label if hasattr(agent, 'label') else method_label) if label is None else label
            v = agent.v if hasattr(agent, 'v') else None
            Q = agent.Q if hasattr(agent, 'Q') else None
            policy = agent.policy if hasattr(agent, 'policy') else None
            v2Q = agent.v2Q if hasattr(agent, 'v2Q') else None
            avail_modes = []
            if Q is not None:
                avail_modes.append("Q")
                avail_modes.append("v")
            elif v is not None:
                avail_modes.append("v")

        if len(avail_modes) > 0:
            self.view_mode = self.view_mode % len(avail_modes)
            if avail_modes[self.view_mode] == 'v':
                preferred_actions = None

                if v == None:
                    preferred_actions = {}
                    v = {s: Q.max(s) for s in self.mdp.nonterminal_states}
                    for s in self.mdp.nonterminal_states:
                        acts, values = Q.get_Qs(s)
                        preferred_actions[s] = [a for (a, w) in zip(acts, values) if
                                                np.round(w, 2) == np.round(v[s], 2)]

                if v2Q is not None:
                    preferred_actions = {}
                    for s in self.mdp.nonterminal_states:
                        q = v2Q(s)
                        mv = np.round(max(q.values()), 2)
                        preferred_actions[s] = [k for k, v in q.items() if np.round(v, 2) == mv]

                self.viewer.update_v(v=v, preferred_actions=preferred_actions, state=state, message=label)

            elif avail_modes[self.view_mode] == 'Q':
                self.viewer.update_q(Q, state=state, message=label)
            else:
                raise Exception("No view mode selected")
        else:
            self.viewer.update_null(state=state, message=label)
            # self.display.displayNullValues(self.mdp, currentState=state)

        # self.display.end_frame()
        render_out = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return render_out


class PygBookGridEnvironment(GridworldPyglet):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_book_grid, *args, **kwargs)


from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.ex11.q_agent import QAgent

if __name__ == "__main__":
    env = PygBookGridEnvironment(zoom=1)

    # env = BookGridEnvironment()
    from irlc import PlayWrapper, VideoMonitor, train, Agent

    agent = QAgent(env)
    # agent = Agent(env)

    # agent = PlayWrapper(agent, env)
    import time
    t0 = time.time()
    n = 500
    env = VideoMonitor(env, agent=agent, fps=1000)
    train(env,agent,num_episodes=1000, max_steps=n, verbose=False)
    tpf = (time.time()-t0)/n
    print("Time per frame", tpf, "fps", 1/tpf)

    env.close()
