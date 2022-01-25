# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import GymPacmanEnvironment
from irlc.project1.pacman import east

count = """
%%%%
%P %
%..%
%%%%
"""

if __name__ == "__main__":
    # Example interaction with an environment:
    # Instantiate the map 'east' and get a GameState instance: 
    env = GymPacmanEnvironment(layout_str=east)
    x = env.reset() # x is not a GameState object. See irlc/pacman/gamestate.py for the definition if you are curious.
    print("Start configuration of board:")
    print(x)
    # The GameState object `x` has a handful of useful functions. These are
    # x.A()       # Action space
    # x.f(action) # State resulting in taking action 'action' in state 'x'
    # x.players() # Number of agents on board (at least 1)
    # x.player()  # Whose turn it is (player = 0 is us)
    # x.isWin()   # True if we have won
    # x.isLose()  # True if we have lost
    # You can check if two GameState objects x1 and x2 are the same by simply doing x1 == x2. 
    # There are other functions in the GameState class, but I advice against using them.

    from irlc import VideoMonitor
    env = VideoMonitor(env)
    env.savepdf('pacman_east')
    env.close()
