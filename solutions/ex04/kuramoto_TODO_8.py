        next_x, cost, terminated, _, metadata = env.step([u]) 
        xs_step.append(next_x)
        ts_step.append(env.time)
        xs_euler.append(dmodel.f( xs_euler[-1], [u], 0))

        if terminated:
            break 