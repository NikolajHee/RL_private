# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc import VideoMonitor, train, Agent, PlayWrapper
from irlc.ex03.pacman_problem_foodsearch import GymFoodSearchProblem
from irlc.ex03.pacman_problem_foodsearch_astar import foodHeuristic
from irlc.ex03.pacman_problem_positionsearch import GymPositionSearchProblem
from irlc.ex03.pacsearch_agents import AStarAgent
from irlc.pacman.devel.new_pacman_environment import NewPacmanEnvironment
from irlc.ex01.agent import train
from irlc.berkley.p4ghostbusters.gym_inference_agent import BusterInferenceAgent, GreedyBustersAgent
from irlc.utils.video_monitor import VideoMonitor
from irlc.pacman import layout
import warnings
import os
from irlc.berkley.p4ghostbusters.busters import BustersGameRules

class NewBustersEnvironment(NewPacmanEnvironment):
    def __init__(self, *args, layout='oneHunt', **kwargs):
        super().__init__(*args, layout='oneHunt', **kwargs)
        self.rules = BustersGameRules()
        self.options_showGhosts = True
        self.first_person_graphics = True


def q1_probabilities():
    warnings.warn("missing unit testing")
    # https://inst.eecs.berkeley.edu/~cs188/su19/project4/

def loadq(q=2,n=1, animate_movement=False, zoom=2.0, BusterAgent=None, env_args={}):
    from irlc.berkley.p4ghostbusters import busters
    # f = busters.__file__

    f = f"{os.path.dirname(busters.__file__)}"

    list = os.listdir(f + f"/test_cases/q{q}/")
    qf = [l for l in list if l.startswith(f"{n}") and l.endswith(".test")].pop()
    with open(f + f"/test_cases/q{q}/{qf}", 'r') as f:
        s = f.read()
    tokens = []
    tk = []
    for l in s.splitlines():
        if ":" in l and len(tk) > 0:
            tokens.append(tk)
            tk = []
        tk.append(l)
    tokens.append(tk)

    tokens = ["\n".join(t) for t in tokens]
    tokens = {t[:t.find(":")].strip(): eval( t[t.find(":")+1:].strip() ) for t in tokens }
    numGhosts = int(tokens['numGhosts'])
    from irlc.berkley.p4ghostbusters.inference import ExactInference, MarginalInference, ParticleFilter
    if tokens['inference'] == "ExactInference":
        InferenceType = ExactInference
    elif tokens['inference'] == 'ParticleFilter':
        InferenceType = ParticleFilter
    elif tokens['inference'] == 'MarginalInference':
        InferenceType = MarginalInference
    else:
        print(tokens['inference'])
        raise Exception

    elapse = tokens['elapse'] == 'True'
    observe = tokens['observe'] == 'True'
    if 'layout_str' in tokens:
        tokens['layout'] = tokens['layout_str']

    from irlc.pacman.pacman_utils import RandomGhost
    from irlc.berkley.p4ghostbusters.tracking_fa18TestClasses import GoSouthAgent, DispersingSeededGhost

    print(f"[Inference] {tokens['inference']}")
    if "ghost" not in tokens:
        tokens['ghost'] = 'SeededRandomGhostAgent'

    print(f"[Ghost] {tokens['ghost']}")
    if tokens['ghost'] == 'DispersingSeededGhost':
        GhostAgent = DispersingSeededGhost
    elif tokens['ghost'] == 'GoSouthAgent':
        GhostAgent = GoSouthAgent
    else:
        GhostAgent = RandomGhost

    env = NewBustersEnvironment(num_ghosts=numGhosts, animate_movement=animate_movement, ghost_agent=GhostAgent, zoom=zoom, **env_args)
    env.layout = layout.Layout(tokens['layout'].strip().splitlines())

    if BusterAgent is None:
        BusterAgent = BusterInferenceAgent
    # else:


    agent = BusterAgent(env, inferenceType=InferenceType, elapseTimeEnable=elapse, observeEnable=observe)
    env = VideoMonitor(env2=env, agent=agent)
    train(env, agent, num_episodes=1, max_steps=int(tokens['maxMoves']))

    env.close()
    a = 234
    pass


def pacman_ghostbeliefs():
    loadq(q=4, n=2)
    pass

def pacman_labyrinth():
    def maze_search(layout='tinyMaze', SAgent=None, heuristic=None, problem=None, render=True, zoom=1.0):
        if problem is None:
            problem = GymPositionSearchProblem()
        env = NewPacmanEnvironment(layout=layout, zoom=zoom, animate_movement=render)
        if heuristic is None:
            agent = SAgent(env, problem=problem)
        else:
            agent = SAgent(env, problem=problem, heuristic=heuristic)

        if render:
            env = VideoMonitor(env, agent=agent, agent_monitor_keys=())
        stats, trajectory = train(env, agent, num_episodes=1, verbose=False, return_trajectory=True)
        reward = stats[0]['Accumulated Reward']
        length = stats[0]['Length']
        print(f"Environment terminated in {length} steps with reward {reward}\n")
        env.close()
    render = True

    ### A^* search and a heuristics function
    maze_search(layout='trickySearch', SAgent=AStarAgent, heuristic=foodHeuristic, problem=GymFoodSearchProblem(),
                render=render, zoom=2)

def pacman_base():
    env = NewPacmanEnvironment(layout='MediumClassic', zoom=2, animate_movement=True)
    agent = Agent(env)
    agent = PlayWrapper(agent, env)
    env = VideoMonitor(env, agent)
    train(env, agent, max_steps=100, verbose=False)
    env.close()


if __name__ == "__main__":
    # pacman_ghostbeliefs()
    pacman_labyrinth()
    # pacman_base()
