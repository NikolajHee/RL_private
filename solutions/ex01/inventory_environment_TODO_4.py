        a = agent.pi(s, k)
        sp, r, done, metadata = env.step(a)
        agent.train(s, a, sp, r, done)
        s = sp
        J += r
        if done:
            break 