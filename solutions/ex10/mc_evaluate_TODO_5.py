        agent = MCEvaluationAgent(env, gamma=gamma)
        agent_every = MCEvaluationAgent(env, gamma=gamma, first_visit=False)

        train(env, agent, num_episodes=episodes, verbose=False)
        train(env, agent_every, num_episodes=episodes, verbose=False)

        ev.append(np.mean(list(agent_every.v.values())))
        fv.append(np.mean(list(agent.v.values()))) 