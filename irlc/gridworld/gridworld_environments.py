# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf).
"""
import numpy as np
from collections import defaultdict
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from gym.spaces.discrete import Discrete
from irlc.ex09.mdp import MDP2GymEnv
from irlc.gridworld.gridworld_mdp import GridworldMDP, FrozenGridMDP
from irlc import Timer
from gym.spaces.multi_discrete import MultiDiscrete
import pygame
# import matplotlib
# matplotlib.use('TkAgg')

grid_cliff_grid = [[' ',' ',' ',' ',' ', ' ',' ',' ',' ',' ', ' '],
                   [' ',' ',' ',' ',' ', ' ',' ',' ',' ',' ', ' '],
                   ['S',-100, -100, -100, -100,-100, -100, -100, -100, -100, 0]]

grid_cliff_grid2 = [[' ',' ',' ',' ',' '],
                    ['S',' ',' ',' ',' '],
                     [-100,-100, -100, -100, 0]]

grid_discount_grid = [[' ',' ',' ',' ',' '],
                    [' ','#',' ',' ',' '],
                    [' ','#', 1,'#', 10],
                    ['S',' ',' ',' ',' '],
                    [-10,-10, -10, -10, -10]]

grid_bridge_grid = [[ '#',-100, -100, -100, -100, -100, '#'],
        [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
        [ '#',-100, -100, -100, -100, -100, '#']]

grid_book_grid = [[' ',' ',' ',+1],
        [' ','#',' ',-1],
        ['S',' ',' ',' ']]

grid_maze_grid = [[' ',' ',' ', +1],
                  ['#','#',' ','#'],
                  [' ','#',' ',' '],
                  [' ','#','#',' '],
                  ['S',' ',' ',' ']]

sutton_corner_maze = [[  1, ' ', ' ', ' '], 
                      [' ', ' ', ' ', ' '],
                      [' ', 'S', ' ', ' '],
                      [' ', ' ', ' ',   1]] 

# A big yafcport open maze.
grid_open_grid = [[' ']*8 for _ in range(5)]
grid_open_grid[0][0] = 'S'
grid_open_grid[-1][-1] = 1


class GridworldEnvironment(MDP2GymEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1000,
    }
    def get_keys_to_action(self):
        return {(pygame.K_LEFT,): GridworldMDP.WEST, (pygame.K_RIGHT,): GridworldMDP.EAST,
                (pygame.K_UP,): GridworldMDP.NORTH, (pygame.K_DOWN,): GridworldMDP.SOUTH}

        # return {(key.LEFT,): GridworldMDP.WEST, (key.RIGHT,): GridworldMDP.EAST, (key.UP,): GridworldMDP.NORTH, (key.DOWN,): GridworldMDP.SOUTH}

    def _get_mdp(self, grid, uniform_initial_state=False):
        return GridworldMDP(grid, living_reward=self.living_reward)

    def __init__(self, grid=None, uniform_initial_state=True, living_reward=0,zoom=1, view_mode=0, render_mode=None, **kwargs):
        self.living_reward = living_reward
        mdp = self._get_mdp(grid)
        self.render_mode = render_mode
        super().__init__(mdp, render_mode=render_mode)
        self.action_space = Discrete(4)
        # self.observation_space = MultiDiscrete([mdp.height, mdp.width]) # N.b. the state space does not contain the terminal state.
        self.render_episodes = 0
        self.render_steps = 0
        self.timer = Timer()
        self.view_mode = view_mode
        self.agent = None # If this is set, the environment will try to render the internal state of the agent.
                          # It is a little hacky, it allows us to make the visualizations etc.
        # Set up rendering if required.
        self.display_pygame = None
        # self.screen = None
        self.zoom = zoom # Save zoom level.

        # self.display_pygame.autorefresh(self, interval=0.1)
        #
        # # main loop
        # while True:
        #     # handle user input
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             import sys
        #             sys.exit()



        def _step(*args, **kwargs):
            o = type(self).step(self, *args, **kwargs)
            done = o[2]
            self.render_steps += 1
            self.render_episodes += done
            # print(o[-1])
            # if 'mask' in o[-1] and o[-1] is None:
            #     print("also bad.")
            # if 'mask' in o[-1] and o[-1]['mask'].max() > 1:
            #     print("BAd!")

            return o
        self.step = _step

    def keypress(self, key):
        if key.unicode == 'm':
            # changing mode...
            self.view_mode += 1
            self.render()
            return

        if key == 116:  # This may easily not be used.
            self.view_mode += 1
            self.render()


    def render(self):
        print("rendering")
        if self.display_pygame is None:
            # self.display = None
            # self.viewer = None
            # speed = 1
            # gridSize =
            # self.display = gridworld_graphics_display.GraphicsGridworldDisplay(self.mdp, gridSize)
            from irlc.gridworld.gridworld_graphics_display import GraphicsGridworldDisplay
            self.display_pygame = GraphicsGridworldDisplay(self.mdp, size=int(150 * self.zoom)) # last item is grid size
            print("I made a display")

        # print("Render callsed")
        # if Q is not None:
        #     raise Exception("Oh dear")
        # if v is not None:
        #     raise Exception("Oh no!")

        # self.agent = agent
        # if label is None:
        #     label = f"{method_label} AFTER {self.render_steps} STEPS"
        agent = self.agent
        label = None
        method_label = ""
        if label is None:
            label = f"{method_label} AFTER {self.render_steps} STEPS"

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
                        preferred_actions[s] = [a for (a,w) in zip(acts, values) if np.round(w, 2) == np.round(v[s], 2)]

                if v2Q is not None:
                    preferred_actions = {}
                    for s in self.mdp.nonterminal_states:
                        q = v2Q(s)
                        mv = np.round( max( q.values() ), 2)
                        preferred_actions[s] = [k for k, v in q.items() if np.round(v, 2) == mv]

                if agent != None and hasattr(agent, 'policy') and agent.policy is not None and state in agent.policy and isinstance(agent.policy[state], dict):
                    for s in self.mdp.nonterminal_states:
                        preferred_actions[s] = [a for a, v in agent.policy[s].items() if v == max(agent.policy[s].values()) ]

                if hasattr(agent, 'returns_count'):
                    returns_count = agent.returns_count
                else:
                    returns_count = None
                if hasattr(agent, 'returns_sum'):
                    returns_sum = agent.returns_sum
                else:
                    returns_sum = None
                self.display_pygame.displayValues(mdp=self.mdp, v=v, preferred_actions=preferred_actions, currentState=state, message=label, returns_count=returns_count, returns_sum=returns_sum)

            elif avail_modes[self.view_mode] == 'Q':

                if hasattr(agent, 'e') and isinstance(agent.e, defaultdict):
                    eligibility_trace = defaultdict(float)
                    for k, v in agent.e.items():
                        eligibility_trace[k] = v

                else:
                    eligibility_trace = None
                self.display_pygame.displayQValues(self.mdp, Q, currentState=state, message=label, eligibility_trace=eligibility_trace)
            else:
                raise Exception("No view mode selected")
        else:
            # self.pygame_display = Gridworl
            self.display_pygame.displayNullValues(self.mdp, currentState=state, message=label)
            # self.display.displayNullValues(self.mdp, currentState=state)

        render_out2 = self.display_pygame.blit(render_mode=self.render_mode)
        print("I blitted")
        # print("> blitted")
        return render_out2
        # if self.render_mode == 'human':
        #
        #
        #     pass
        # self.display.end_frame()
        # render_out = self.viewer.render(return_rgb_array=self.render_mode == 'rgb_array')
        # return render_out

    def close(self):
        print("Closing time...")
        if self.display_pygame is not None:
            self.display_pygame.close()


class BookGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_book_grid, *args, **kwargs)

class BridgeGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_bridge_grid, *args, **kwargs)

class CliffGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_cliff_grid, living_reward=-1, *args, **kwargs)

class CliffGridEnvironment2(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_cliff_grid2, living_reward=-1, *args, **kwargs)


class OpenGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_open_grid, *args, **kwargs)

"""  
Implement Suttons little corner-maze environment (see (SB18, Example 4.1)).  
You can make an instance using:
> from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
> env = SuttonCornerGridEnvironment()
To get access the the mdp (as a MDP-class instance, for instance to see the states env.mdp.nonterminal_states) use
> env.mdp
You can make a visualization (allowing you to play) and train it as:
> from irlc import PlayWrapper, Agent, train
> agent = Agent()
> agent = PlayWrapper(agent, env) # allows you to play using the keyboard; omit to use agent as usual. 
> train(env, agent, num_episodes=1)
"""
class SuttonCornerGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, living_reward=-1, **kwargs): # living_reward=-1 means the agent gets a reward of -1 per step.
        super().__init__(sutton_corner_maze, *args, living_reward=living_reward, **kwargs) 

class SuttonMazeEnvironment(GridworldEnvironment):
    def __init__(self, *args, render_mode=None, living_reward=0, **kwargs):
        sutton_maze_grid = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', '#',  +1],
                            [' ', ' ', '#', ' ', ' ', ' ', ' ', '#', ' '],
                            ['S', ' ', '#', ' ', ' ', ' ', ' ', '#', ' '],
                            [' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' '],
                            [' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' '],
                            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]

        super().__init__(sutton_maze_grid, *args, render_mode=render_mode, living_reward=living_reward, **kwargs)




# "4x4":[
#     "SFFF",
#     "FHFH",
#     "FFFH",
#     "HFFG"
#     ]

# "8x8": [
#     "SFFFFFFF",
#     "FFFFFFFF",
#     "FFFHFFFF",
#     "FFFFFHFF",
#     "FFFHFFFF",
#     "FHHFFFHF",
#     "FHFFHFHF",
#     "FFFHFFFG",
# ]
# frozen_lake_4 = [['S', ' ', ' ', ' '],
#                  [' ',   0, ' ',   0],
#                  [' ', ' ', ' ',   0],
#                  [  0, ' ', ' ',  +1]]

grid_book_grid_ = [[' ',' ',' ',+1],
                  [' ','#',' ',-1],
                  ['S',' ',' ',' ']]

frozen_lake_4 = [['S',' ',' ',' '],
                 [' ','#',' ',-1],
                 [ 0 , ' ', ' ',  +1]]
# Single letter in front: 0 -> ' ' fix the problem. WHAT THE FUCK IS WRONG WITH THIS????

# class FrozenLake2(GridworldEnvironment):
#     def __init__(self, *args, **kwargs):
#         super().__init__(frozen_lake_4, *args, **kwargs)

class FrozenLake(GridworldEnvironment):
    def _get_mdp(self, grid, uniform_initial_state=False):
        return FrozenGridMDP(grid, is_slippery=self.is_slippery, living_reward=self.living_reward)

    def __init__(self, is_slippery=True, living_reward=0, *args, **kwargs):
        self.is_slippery = is_slippery
        menv = FrozenLakeEnv(is_slippery=is_slippery) # Load frozen-lake game layout and convert to our format 'grid'
        gym2grid = dict(F=' ', G=1, H=0)
        grid = [[gym2grid.get(s.decode("ascii"), s.decode("ascii")) for s in l] for l in menv.desc.tolist()]
        # grid =  frozen_lake_4
        menv.close()
        super().__init__(grid=grid, *args, living_reward=living_reward, **kwargs)

if __name__ == "__main__":
    import gym
    # env = gym.make('CartPole-v1', render_mode="human")
    # env.reset()
    #
    # a = 234 gym
    # env = gym.make('CartPole-v1', render_mode="human")
    # env.reset()
    from irlc import interactive, Agent, train
    from irlc.ex11.q_agent import QAgent
    from irlc.ex11.sarsa_agent import SarsaAgent
    # env = SuttonMazeEnvironment(render_mode="human", zoom=0.75)
    # env = OpenGridEnvironment(render_mode='human', zoom=0.75)
    # env = OpenGridEnvironment()
    env = CliffGridEnvironment()
    agent = QAgent(env)
    # env, agent = interactive(env, QAgent(env))
    # stats, trajectories = train(env, agent, num_episodes=100, experiment_name='q_learning')
    stats, trajectories = train(env, SarsaAgent(env), num_episodes=100, experiment_name='sarsa')

    from irlc import main_plot
    main_plot(experiments=['q_learning', 'sarsa'])
    from matplotlib import pyplot as plt
    plt.show()
    # from irlc import VideoMonitor, train, Agent, PlayWrapper
    # agent = Agent(env)
    env.reset()
    env.close()

    # agent = PlayWrapper(agent, env)
    # env = VideoMonitor(env)
    # env = Video

    # a = 234
    # for r in range(100):
    #     import time
    #     env.reset()
    # time.sleep(1)
    # train(env, agent, 2000)
    a = 234
    # env.step(0)
