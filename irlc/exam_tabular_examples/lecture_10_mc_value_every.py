# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.lectures.lec10.lecture_10_mc_q_estimation import keyboard_play_value
from irlc.gridworld.gridworld_environments import BookGridEnvironment
from irlc.ex10.mc_evaluate import MCEvaluationAgent

if __name__ == "__main__":
    env = BookGridEnvironment(view_mode=1)
    agent = MCEvaluationAgent(env, gamma=.9, alpha=None, first_visit=False)
    keyboard_play_value(env,agent,method_label='MC every')