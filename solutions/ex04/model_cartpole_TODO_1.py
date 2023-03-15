#     
#     bounds = dict(tF_low=0.01, tF_high=np.inf,
#                   x_low=[-2 * dist, -np.inf, -2 * np.pi, -np.inf], x_high=[2 * dist, np.inf, 2 * np.pi, np.inf],
#                   u_low=[-maxForce], u_high=[maxForce],
#                   x0_low=[0, 0, np.pi, 0], x0_high=[0, 0, np.pi, 0],
#                   xF_low=[dist, 0, 0, 0], xF_high=[dist, 0, 0, 0])  