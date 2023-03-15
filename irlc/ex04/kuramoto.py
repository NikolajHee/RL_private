# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her23] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2023.
"""
import sympy as sym
from irlc.ex04.continuous_time_model import ContiniousTimeSymbolicModel
from irlc.ex04.cost_continuous import SymbolicQRCost
from irlc.ex04.continuous_time_discretized_model import DiscretizedModel
from irlc.ex04.continuous_time_environment import ContiniousTimeEnvironment
import numpy as np
from irlc import train, Agent, savepdf


class ContiniousKuramotoModel(ContiniousTimeSymbolicModel): # same role as DPModel
    def __init__(self): 
        """
        Create a cost-object. The code defines a quadratic cost (with the given matrices) and allows easy computation
        of derivatives, etc. There are automatic ways to discretize the cost so you don't have to bother with that.
        See the online documentation for further details.
        """
        cost = SymbolicQRCost(R=np.ones( (1,1) ), Q=np.zeros((1,1))) 
        """ Defines the simple bounds. In this case the system is constrained to start in x(0) = 0 and -2<= u(t) <= 2."""
        bounds = dict(x_low=[-np.inf],  x_high=[np.inf], # No non-boundary restrictions
                      u_low=[-2],       u_high=[2],      # -2 <= u <= 2
                      x0_low=[0],       x0_high=[0])     # x(0) = 0 (initial state). 
        """This call is important: This is where the symbolic function self.sym_f is turned into a numpy function 
        automatically. Although what happens in the super class is a little more complicated than most of the code we 
        have seen so far, I still recommend looking at it for reference."""
        super().__init__(cost=cost, bounds=bounds) 

    def sym_f(self, x, u, t=None): 
        """ Return a symbolic expression representing the Kuramoto model.
        The inputs x, u are themselves *lists* of symbolic variables (insert breakpoint and check their value).
        you have to use them to create a symbolic object representing f, and return it as a list. That is, you are going to return
        > return [f(x,u)]
        where f is the symbolic expression. Note you can use trigonometric functions like sym.cos. 
        """
        # TODO: 1 lines missing.
        #raise NotImplementedError("Implement symbolic expression as a singleton list here")
        #breakpoint()
        symbolic_f_expression  = [sym.N(uk + sym.cos(xk)) for (uk,xk) in zip(u,x)]
        # define the symbolic expression 
        return symbolic_f_expression  

class DiscreteKuramotoModel(DiscretizedModel): 
    """ Create a discrete version of the Kuramoto environment.
    The superclass will automatically Euler discretize the continuous model (time constant 0.5) and set up useful functionality.
    """
    def __init__(self, dt=0.5):
        model = ContiniousKuramotoModel()
        super().__init__(model=model, dt=dt) 

class KuramotoEnvironment(ContiniousTimeEnvironment): 
    """ Turn the whole thing into an environment. The step()-function in the environment will use *exact* RK4 simulation.
    and automatically compute the cost using the cost-function. Tmax is the total simulation time in seconds.
    """
    def __init__(self, Tmax=5, dt=0.5):
        discrete_model = DiscreteKuramotoModel(dt)
        super().__init__(discrete_model, Tmax=Tmax) 


def f(x, u):
    """ Implement the kuramoto osscilator model's dynamics, i.e. f such that dx/dt = f(x,u).
    The answer should be returned as a singleton list. """
    cmodel = ContiniousKuramotoModel()
    # TODO: 1 lines missing.
    #raise NotImplementedError("Insert your solution and remove this error.")
    # Use the ContiniousKuramotoModel to compute f(x,u). If in doubt, insert a breakpoint and let pycharms autocomplete
    # guide you. See my video to Exercise 2 for how to use the debugger. Don't forget to specify t (for instance t=0).
    # Note that sympys error messages can be a bit unforgiving.
    f_value = cmodel.sym_f(x,u,t=0)
    
    return f_value

def fk(x,u):
    """ Computes the discrete (Euler 1-step integrated) version of the Kuromoto update with discretization time dt=0.5,i.e.

    x_{k+1} = f_k(x,u).

    Look at dmodel.f for inspiration. As usual, use a debugger and experiment. Note you have to specify input arguments as lists,
    and the function should return a numpy ndarray.
    """
    dmodel = DiscreteKuramotoModel(dt=0.5)
    # TODO: 1 lines missing.
    #raise NotImplementedError("Compute Euler discretized dynamics here using the dmodel.")
    f_euler = dmodel.f(x,u)
    return f_euler

def dfk_dx(x,u):
    """ Computes the derivative of the (Euler 1-step integrated) version of the Kuromoto update with discretization time dt=0.5,
    i.e. if

    x_{k+1} = f_k(x,u)

    this function should return

    df_k/dx

    (i.e. the Jacobian with respect to x) as a numpy matrix.
    Look at dmodel.f for inspiration, and note it has an input argument that is relevant.
    As usual, use a debugger and experiment. Note you have to specify input arguments as lists,
    and the function should return a two-dimensional numpy ndarray.

    """
    dmodel = DiscreteKuramotoModel(dt=0.5)
    # the function dmodel.f accept various parameters. Perhaps their name can give you an idea?
    # TODO: 1 lines missing.
    _, Jx, Ju = dmodel.f(x,u,compute_jacobian=True)
    #raise NotImplementedError("Compute derivative here using the dmodel.")
    return Jx

def rk4_simulate(x0, u, t0, tF, N=1000):
    """
    Implement the RK4 algorithm (Her23, Algorithm 18).
    In this function, x0 and u are constant numpy ndarrays. I.e. u is not a function, which simplify the RK4
    algorithm a bit.

    The function you want to integrate, f, is already defined above. You can likewise assume f is not a function of
    time. t0 and tF play the same role as in the algorithm.

    The function should return a numpy ndarray xs of dimension (N,) (containing all the x-values) and a numpy ndarray
    tt containing the corresponding time points.

    Hints:
        * Call f as in f(x, u). You defined f earlier in this exercise.
    """
    tt = np.linspace(t0, tF, N+1)   # Time grid t_k = tt[k] between t0 and tF.
    xs = [ x0 ]
    f(x0, u) # This is how you can call f.

    delta = (tF - t0)/N
    for k in range(N):
        x_next = None # Obtain x_next = x_{k+1} using a single RK4 step.
        # Remember to insert breakpoints and use the console to examine what the various variables are.
        # TODO: 7 lines missing.

        k1 = np.asarray(f(xs[-1], u))
        k2 = np.asarray(f(xs[-1] + delta*k1/2, u))
        k3 = np.asarray(f(xs[-1] + delta*k2/2, u))
        k4 = np.asarray(f(xs[-1] + delta*k3, u))

        x_next = xs[-1] + 1/6 * delta *( k1 + 2*k2 + 2*k3 + k4)

        #raise NotImplementedError("Insert your solution and remove this error.")
        xs.append(x_next)
    xs = np.stack(xs, axis=0)
    return xs, tt

if __name__ == "__main__":
    # Part 1: A sympy warmup. This defines a fairly nasty sympy function:
    z = sym.symbols('z')    # Create a symbolic variable 
    g = sym.exp( sym.cos(z) ** 2) * sym.sin(z) # Create a nasty symbolic expression.
    print("z is:", z, " and g is:", g) 
    # TODO: 1 lines missing.
    dg_dz = sym.diff(g, z)
    #raise NotImplementedError("Compute the derivative of g here (symbolically)")
    print("The derivative of the nasty expression is dg/dz =", dg_dz)

    # TODO: 1 lines missing.
    #raise NotImplementedError("Turn the symbolic expression into a function using sym.lambdify. Check the notes for an example (or sympys documentation).")
    g_as_a_function = sym.lambdify(z, dg_dz)
    print("dg/dz (when z=0) =", g_as_a_function(0))
    print("dg/dz (when z=pi/2) =", g_as_a_function(np.pi/2))
    print("(Compare these results with the symbolic expression)") 

    # Part 2: Create a symbolic model corresponding to the Kuramoto model:
    # This is an example of using the model to compute dx/dt = f(x, u, t). To make it work, implement the f-function:
    print("Value of f(x,u) in x=2, u=0.3", f([2], [0.3])) 
    print("Value of f(x,u) in x=0, u=1", f([0], [1])) 

    """We use cmodel.simulate(...) to simulate the environment, starting in x0 =0, from time t0=0 to tF=10, 
    using a constant action of u=1.5. Note u_fun in the simulate function can be set to a constant. Use this 
    compute numpy ndarrays corresponding to the time, x and u values.
    
    To make this work, you have to implement RK4 integration. 
    """
    cmodel = ContiniousKuramotoModel()
    # print(cmodel) # Uncomment this line to print details about the environment.
    x0 = cmodel.bounds['x0_low'] # Get the starting state x0. We exploit that the bound on x0 is an equality constraint.
    u = 1.3
    xs, ts = rk4_simulate(x0, [u], t0=0, tF=20, N=100)
    xs_true, us_true, ts_true = cmodel.simulate(x0, u_fun=u, t0=0, tF=20, N_steps=100)

    # # Plot the exact simulation of the environment
    import matplotlib.pyplot as plt
    # plt.plot(ts_true, xs_true, 'k.-', label='RK4 state sequence x(t) (using model.simulate)')
    # plt.plot(ts, xs, 'r-', label='RK4 state sequence x(t) (using your code)')
    # plt.legend()
    # savepdf('kuramoto_rk4')
    # plt.show()


    # plt.figure(2)
    # Part 3: The discrete environment
    dmodel = DiscreteKuramotoModel()  # Create a *discrete* model
    print(dmodel) # Uncomment this line to see details about the environment.

    print("The Euler-discretized version, f_k(x,u) = x + Delta f(x,u), is") 
    print("f_k(x=0,u=0) =", fk([0], [0]))
    print("f_k(x=1,u=0.3) =", fk([1], [0.3]))

    print("The derivative of the Euler discretized version wrt. x is:")
    print("df_k/dx(x=0,u=0) =", dfk_dx([0], [0])) 



    # Part 4: The environment and simulation:

    env = KuramotoEnvironment(Tmax=20)  # An environment that runs for 5 seconds.

    ts_step = []  # Current time (according to the environment, i.e. in increments of dt.
    xs_step = []  # x_k using the env.step-function in the enviroment.
    xs_euler = [] # x_k using Euler discretization.

    x, _ = env.reset()       # Get starting state.
    ts_step.append(env.time) # env.time keeps track of the clock-time in the environment.
    xs_step.append(x)        # Initialize with first state
    xs_euler.append(x)       # Initialize with first state

    # Use
    # > next_x, cost, terminated, truncated, metadata = env.step([u])
    # to simulate a single step.
    for _ in range(10000):
        # TODO: 7 lines missing.
        xp, reward, terminate, _, info = env.step([u])
        xs_step.append(xp)

        ts_step.append(env.time)
        xs_euler.append(dmodel.f( xs_euler[-1], [u], 0))
        
        if terminate: break


        #raise NotImplementedError("Use the step() function to simulate the environment. Note that the step() function uses RK4.")

    plt.plot(ts, xs, 'k-', label='RK4 (nearly exact)')
    plt.plot(ts_step, xs_step, 'b.', label='RK4 (step-function in environment)')
    plt.plot(ts_step, xs_euler, 'r.', label='Euler (dmodel.f(last_x, action, k)')

    # Train and plot a random agent.
    env = KuramotoEnvironment(Tmax=20) 
    stats, trajectories = train(env, Agent(env), return_trajectory=True)
    plt.plot(trajectories[0].time, trajectories[0].state, label='x(t) when using a random action sequence from agent') 
    plt.legend()
    savepdf('kuramoto_step')
    plt.show()
    print("The total cost obtained using random actions", -stats[0]['Accumulated Reward'])
