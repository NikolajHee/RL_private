    stats, trajectories = train(env, FrozenAgentDownRight(env), num_episodes=50, return_trajectory=True) 
    for trajectory in trajectories:
        p = [to_rc(s, ncol=env.env.ncol) for s in trajectory.state]
        I, J = zip(*p)
        wgl = 0.1
        plt.plot(np.random.randn(len(I)) * wgl + I, np.random.randn(len(J)) * wgl + J, 'k.-') 