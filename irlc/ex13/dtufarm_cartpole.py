# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import inspect
import sys
import os
# import pickle
CDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.append(CDIR + "/../../../../thtools")
sys.path.append(CDIR + "/../../")
# from thtools.farm import autogrid
# import thtools

# # <grid_rm>
# if True:
#     script = "farm_keras_cartpole.py"
#     if thtools.is_win():
#         through_compute_conf = {'through_compute': True,
#                                 'git_remote': ['~/Documents/02465public'],
#                                 'git_local': ['C:/Users/tuhe/Documents/02465public'],
#                                 'remote_file': f'~/Documents/02465public/pythontools/irlc/ex13/{script}'
#                                 }
#     else:
#         through_compute_conf = None
#     # venv = "tf12" # use with double Q code.
#     venv = "tf21" # use with dueling q code
#     rs = autogrid(script=script, rs="rs", virtual_environment_name=venv, opts={}, max_running=12,
#                   rs_out="example_out.pkl", through_compute_conf=through_compute_conf)
#
#
#     with open("example_out.pkl", 'rb') as f:
#         s = pickle.load(f)
#         print("Results on " + thtools.get_system_name())
#         print(s)
#         thtools.pprint(s)
#     sys.exit()
# # </grid_rm>

def grid_sim(k, env="cartpole", method="dqn", experiment_name="unnamed_experiment", num_episodes=200):
    if method == 'dqn':
        # if c == 0:  # DQN
        #     name = "dqn"
        from irlc.ex13.deepq_agent import mk_cartpole
        env, agent = mk_cartpole()
    elif method == 'double_dqn':
        from irlc.ex13.double_deepq_agent import mk_cartpole
        env, agent = mk_cartpole()
        pass
    elif method == 'duel_dqn':
        from irlc.ex13.duel_deepq_agent import mk_cartpole
        env, agent = mk_cartpole()
    else:
        raise Exception("bad method given", method)

    # if c == 1:  # double DQN
    # name = "double_dqn"
    # if c == 2:  # duel DQN
    #     name = "duel_dqn"
    # env, agent = mkfun()
    from irlc.ex01.agent import train
    train(env, agent, experiment_name=experiment_name, num_episodes=num_episodes)
    return "ok"

# rs = dict()
# # max_iter=200
# for k in range(4):
#     for c in range(3):
#         # <grid_fun>
#         env = "cartpole"
#         if c == 0: # DQN
#             name = "dqn"
#             from irlc.ex13.deepq_agent import mk_cartpole as mkfun
#         if c == 1: # double DQN
#             from irlc.ex13.double_deepq_agent import mk_cartpole as mkfun
#             name = "double_dqn"
#         if c == 2: # duel DQN
#             from irlc.ex13.duel_deepq_agent import mk_cartpole as mkfun
#             name = "duel_dqn"
#         ex = f"exps/{env}_{name}_C"
#         grid_sim(mkfun, experiment_name=ex, num_episodes=200)
#         rs[(k,c)] = (k, c, "2020_tuesday")
#         # </grid_fun>

if __name__ == "__main__":
    from dtufarm.farm import DTUCluster
    restart = False
    with DTUCluster(job_group="jobs/job0", nuke_all_remote_folders=restart, nuke_group_folder=restart,
                    dir_map=["../../irlc"], track_jobs_after_submission=False, rq=False, disable=False,
                    ) as dc:
        dc.executor = None
        gs = dc.wrap(grid_sim)
        rs = {}
        for k in range(4): # should eventually be 4.
            for method in ['dqn', 'double_dqn', 'duel_dqn']:
                # for c in range(3):
                env = "cartpole"
                ex = f"exps/{env}_{method}_D"
                rs[(k,method)] = gs(k=k, env=env, method=method, experiment_name=ex, num_episodes=200)
                # gs(mkfun, experiment_name=ex, num_episodes=200)
                # rs[(k, c)] = (k, c, "2020_tuesday")

        a = 234

""" See
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://github.com/cyoon1729/deep-Q-networks
https://medium.com/@leosimmons/double-dqn-implementation-to-solve-openai-gyms-cartpole-v-0-df554cd0614d
https://github.com/lsimmons2/double-dqn-cartpole-solution/blob/master/double_dqn.py
"""
