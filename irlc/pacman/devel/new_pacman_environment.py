# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.devel.pyglet_pacman_graphics import PacmanViewer
from irlc.pacman.gamestate import Directions, ClassicGameRules
from irlc.pacman.layout import getLayout
from irlc.pacman.pacman_text_display import PacmanTextDisplay
from irlc.pacman.pacman_utils import PacAgent, RandomGhost
from irlc.pacman.layout import Layout
import gym
from irlc.utils.common import ExplicitActionSpace
from pyglet.window import key


class NewPacmanEnvironment(gym.Env):
    _unpack_search_state = True  # A hacky fix to set the search state.

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self, animate_movement=False, layout='mediumGrid', zoom=2.0, num_ghosts=4, frames_per_second=30,
                 ghost_agent=None, layout_str=None):
        self.metadata['video_frames_per_second'] = frames_per_second
        self.ghosts = [ghost_agent(i + 1) if ghost_agent is not None else RandomGhost(i + 1) for i in range(num_ghosts)]

        # Set up action space. Use this P-construction so action space can depend on state (grr. gym).
        class P:
            def __getitem__(self, state):
                return {pm_action: "new_state" for pm_action in state.A()}

        self.P = P()
        self.action_space = ExplicitActionSpace(self)  # Wrapper environments copy the action space.

        # Load level layout
        if layout_str is not None:
            self.layout = Layout([line.strip() for line in layout_str.strip().splitlines()])
        else:
            self.layout = getLayout(layout)
            if self.layout is None:
                raise Exception("Layout file not found", layout)
        self.rules = ClassicGameRules(30)
        self.options_frametime = 1 / frames_per_second
        self.game = None

        # Setup displays.
        self.first_person_graphics = False
        self.animate_movement = animate_movement
        self.options_zoom = zoom
        self.text_display = PacmanTextDisplay(1 / frames_per_second)
        self.viewer = None

    def reset(self):
        self.game = self.rules.newGame(self.layout, PacAgent(index=0), self.ghosts, quiet=True, catchExceptions=False)
        self.game.numMoves = 0
        return self.state

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def state(self):
        if self.game is None:
            return None
        return self.game.state.deepCopy()

    def get_keys_to_action(self):
        return {(key.LEFT,): Directions.WEST,
                (key.RIGHT,): Directions.EAST,
                (key.UP,): Directions.NORTH,
                (key.DOWN,): Directions.SOUTH}

    def step(self, action):
        r_ = self.game.state.getScore()
        done = False
        if action not in self.P[self.game.state]:
            raise Exception(f"Agent tried {action=} available actions {self.P[self.game.state]}")
        old_state = self.game.state.data

        # Let player play `action`, then let the ghosts play their moves in sequence.
        for agent_index in range(len(self.game.agents)):
            a = self.game.agents[agent_index].getAction(self.game.state) if agent_index > 0 else action
            self.game.state = self.game.state.f(a)
            self.game.rules.process(self.game.state, self.game)

            if self.viewer is not None and self.animate_movement and agent_index == 0:
                self.viewer.animate_pacman(old_state, self.game.state.data)  # Change the display

            done = self.game.gameOver or self.game.state.isWin() or self.game.state.isLose()
            if done:
                break
        reward = self.game.state.getScore() - r_
        return self.state, reward, done, {}

    def render(self, mode='human', agent=None):
        if mode in ["human", 'rgb_array']:
            if self.viewer is None:
                self.viewer = PacmanViewer(self.game.state.data, self.options_zoom, frame_time=self.options_frametime)

            path = agent.path if hasattr(agent, 'path') else None
            ghostbeliefs = agent.ghostbeliefs if hasattr(agent, 'ghostbeliefs') else None
            visitedlist = agent.visitedlist if hasattr(agent, 'visitedlist') else None

            self.viewer.update(self.game.state.data, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)
            return self.viewer.render(return_rgb_array=mode == "rgb_array")
        elif mode in ['ascii']:
            return self.text_display.draw(self.game.state)
        else:
            raise Exception("Bad video mode", mode)
