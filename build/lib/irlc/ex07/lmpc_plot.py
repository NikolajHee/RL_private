# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def plotTrajectory(map, x, x_glob, u):
    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width, epsi=None)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width, epsi=None)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0, epsi=None)

    plt.figure()
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(x_glob[:, 4], x_glob[:, 5], '-r')

    plt.figure()
    plt.subplot(711)
    plt.plot(x[:, 4], x[:, 0], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    plt.plot(x[:, 4], x[:, 1], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    plt.plot(x[:, 4], x[:, 2], '-o')
    plt.ylabel('wz')
    plt.subplot(714)
    plt.plot(x[:, 4], x[:, 3], '-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    plt.plot(x[:, 4], x[:, 5], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    plt.plot(x[0:-1, 4], u[:, 0], '-o')
    plt.ylabel('steering')
    plt.subplot(717)
    plt.plot(x[0:-1, 4], u[:, 1], '-o')
    plt.ylabel('acc')

def plotClosedLoopLMPC(LMPController, map):
    SS_glob = LMPController.SSset.SS_glob
    TimeSS  = LMPController.SSset.TimeSS
    SS      = LMPController.SSset.SS
    uSS     = LMPController.SSset.uSS

    TotNumberIt = LMPController.it
    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.figure(1)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')

    for i in range(2, TotNumberIt):
        plt.plot(SS_glob[0:TimeSS[i], 4, i], SS_glob[0:TimeSS[i], 5, i], '-r')

    plt.figure(2)
    plt.subplot(711)
    for i in range(2, TotNumberIt):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 0, i], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    for i in range(2, TotNumberIt):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 1, i], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    for i in range(2, TotNumberIt):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 2, i], '-o')
    plt.ylabel('wz')
    plt.subplot(714)
    for i in range(2, TotNumberIt):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 3, i], '-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    for i in range(2, TotNumberIt):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 5, i], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    for i in range(2, TotNumberIt):
        plt.plot(uSS[0:TimeSS[i] - 1, 0, i], '-o')
    plt.ylabel('Steering')
    plt.subplot(717)
    for i in range(2, TotNumberIt):
        plt.plot(uSS[0:TimeSS[i] - 1, 1, i], '-o')
    plt.ylabel('Acc')


def animation_xy(map, LMPCOpenLoopData, LMPController, it):
    SS_glob = LMPController.SSset.SS_glob
    TimeSS = LMPController.SSset.TimeSS
    # SS = LMPController.SS
    # uSS = LMPController.uSS

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.figure(200)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(SS_glob[0:TimeSS[it], 4, it], SS_glob[0:TimeSS[it], 5, it], '-ok', label="Closed-loop trajectory",zorder=-1)

    ax = plt.axes()
    SSpoints_x = []; SSpoints_y = []
    xPred = []; yPred = []
    SSpoints, = ax.plot(SSpoints_x, SSpoints_y, 'sb', label="SS",zorder=0)
    line, = ax.plot(xPred, yPred, '-or', label="Predicted Trajectory",zorder=1)

    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])
    rec = patches.Polygon(v, alpha=0.7,closed=True, fc='r', ec='k',zorder=10)
    ax.add_patch(rec)

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)

    N = LMPController.N
    numSS_Points = LMPController.SSset.numSS_Points
    for i in range(0, int(LMPController.SSset.TimeSS[it])):

        xPred = np.zeros((N+1, 1)); yPred = np.zeros((N+1, 1))
        SSpoints_x = np.zeros((numSS_Points, 1)); SSpoints_y = np.zeros((numSS_Points, 1))

        for j in range(0, N+1):
            xPred[j,0], yPred[j,0]  = map.getGlobalPosition( LMPCOpenLoopData.PredictedStates[j, 4, i, it],
                                                             LMPCOpenLoopData.PredictedStates[j, 5, i, it] )

            if j == 0:
                x = SS_glob[i, 4, it]
                y = SS_glob[i, 5, it]
                psi = SS_glob[i, 3, it]
                l = 0.4; w = 0.2
                car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l*np.cos(psi) + w * np.sin(psi),
                          x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
                car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
                          y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]

        for j in range(0, numSS_Points):
            SSpoints_x[j,0], SSpoints_y[j,0] = map.getGlobalPosition(LMPCOpenLoopData.SSused[4, j, i, it],
                                                                     LMPCOpenLoopData.SSused[5, j, i, it])
        SSpoints.set_data(SSpoints_x, SSpoints_y)

        line.set_data(xPred, yPred)

        rec.set_xy(np.array([car_x, car_y]).T)
        plt.draw()
        plt.pause(0.1)

def animation_states(map, LMPCOpenLoopData, LMPController, it):
    SS_glob = LMPController.SSset.SS_glob
    TimeSS = LMPController.SSset.TimeSS
    SS = LMPController.SSset.SS
    # uSS = LMPController.uSS

    xdata = []; ydata = []
    fig = plt.figure(100)

    axvx = fig.add_subplot(3, 2, 1)
    plt.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 0, it], '-ok', label="Closed-loop trajectory")
    lineSSvx, = axvx.plot(xdata, ydata, 'sb-', label="SS")
    linevx, = axvx.plot(xdata, ydata, 'or-', label="Predicted Trajectory")
    plt.ylabel("vx")
    plt.xlabel("s")

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)

    axvy = fig.add_subplot(3, 2, 2)
    axvy.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 1, it], '-ok')
    lineSSvy, = axvy.plot(xdata, ydata, 'sb-')
    linevy, = axvy.plot(xdata, ydata, 'or-')
    plt.ylabel("vy")
    plt.xlabel("s")

    axwz = fig.add_subplot(3, 2, 3)
    axwz.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 2, it], '-ok')
    lineSSwz, = axwz.plot(xdata, ydata, 'sb-')
    linewz, = axwz.plot(xdata, ydata, 'or-')
    plt.ylabel("wz")
    plt.xlabel("s")

    axepsi = fig.add_subplot(3, 2, 4)
    axepsi.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 3, it], '-ok')
    lineSSepsi, = axepsi.plot(xdata, ydata, 'sb-')
    lineepsi, = axepsi.plot(xdata, ydata, 'or-')
    plt.ylabel("epsi")
    plt.xlabel("s")

    axey = fig.add_subplot(3, 2, 5)
    axey.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 5, it], '-ok')
    lineSSey, = axey.plot(xdata, ydata, 'sb-')
    lineey, = axey.plot(xdata, ydata, 'or-')
    plt.ylabel("ey")
    plt.xlabel("s")

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    axtr = fig.add_subplot(3, 2, 6)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(SS_glob[0:TimeSS[it], 4, it], SS_glob[0:TimeSS[it], 5, it], '-ok')

    SSpoints_x = []; SSpoints_y = []
    xPred = []; yPred = []
    SSpoints_tr, = axtr.plot(SSpoints_x, SSpoints_y, 'sb')
    line_tr, = axtr.plot(xPred, yPred, '-or')

    N = LMPController.N
    numSS_Points = LMPController.SSset.numSS_Points
    for i in range(0, int(TimeSS[it])):

        xPred    = LMPCOpenLoopData.PredictedStates[:, :, i, it]
        SSpoints = LMPCOpenLoopData.SSused[:, :, i, it]

        linevx.set_data(xPred[:, 4], xPred[:, 0]);   axvx.set_title(str(xPred[0, 0]))
        linevy.set_data(xPred[:, 4], xPred[:, 1]);   axvy.set_title(str(xPred[0, 1]))
        linewz.set_data(xPred[:, 4], xPred[:, 2]);   axwz.set_title(str(xPred[0, 2]))
        lineepsi.set_data(xPred[:, 4], xPred[:, 3]); axepsi.set_title(str(xPred[0, 3]))
        lineey.set_data(xPred[:, 4], xPred[:, 5]);   axey.set_title(str(xPred[0, 5]))

        epsiReal = xPred[0, 3]

        lineSSvx.set_data(SSpoints[4,:], SSpoints[0,:])
        lineSSvy.set_data(SSpoints[4,:], SSpoints[1,:])
        lineSSwz.set_data(SSpoints[4,:], SSpoints[2,:])
        lineSSepsi.set_data(SSpoints[4,:], SSpoints[3,:])
        lineSSey.set_data(SSpoints[4,:], SSpoints[5,:])

        xPred = np.zeros((N + 1, 1));yPred = np.zeros((N + 1, 1))
        SSpoints_x = np.zeros((numSS_Points, 1));SSpoints_y = np.zeros((numSS_Points, 1))

        for j in range(0, N + 1):
            xPred[j, 0], yPred[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.PredictedStates[j, 4, i, it],
                                                             LMPCOpenLoopData.PredictedStates[j, 5, i, it])

        for j in range(0, numSS_Points):
            SSpoints_x[j, 0], SSpoints_y[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.SSused[4, j, i, it],
                                                                       LMPCOpenLoopData.SSused[5, j, i, it])

        line_tr.set_data(xPred, yPred)


        vec = np.array([xPred[0, 0], yPred[0, 0]]) - np.array([SS_glob[i, 4, it], SS_glob[i, 5, it]])

        s, ey, epsi, _ = map.getLocalPosition( SS_glob[i, 4, it], SS_glob[i, 5, it], SS_glob[i, 3, it])
        axtr.set_title(str(s)+" "+str(ey)+" "+str(epsi))

        # axepsi.set_title(str(epsiReal)+" "+str(epsi))
        SSpoints_tr.set_data(SSpoints_x, SSpoints_y)

        plt.draw()
        plt.pause(1e-17)

def saveGif_xyResults(map, LMPCOpenLoopData, LMPController, it):
    SS_glob = LMPController.SS_glob
    TimeSS = LMPController.TimeSS
    SS = LMPController.SS
    uSS = LMPController.uSS

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    fig = plt.figure(101)
    plt.ylim((-5, 1.5))
    fig.set_tight_layout(True)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(SS_glob[0:TimeSS[it], 4, it], SS_glob[0:TimeSS[it], 5, it], '-ok', label="Closed-loop trajectory", markersize=1,zorder=-1)

    ax = plt.axes()
    SSpoints_x = []; SSpoints_y = []
    xPred = []; yPred = []
    SSpoints, = ax.plot(SSpoints_x, SSpoints_y, 'sb', label="SS",zorder=0)
    line, = ax.plot(xPred, yPred, '-or', label="Predicted Trajectory",zorder=1)

    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])
    rec = patches.Polygon(v, alpha=0.7,closed=True, fc='r', ec='k',zorder=10)
    ax.add_patch(rec)

    plt.legend(mode="expand", ncol=3)
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=3)

    N = LMPController.N
    numSS_Points = LMPController.numSS_Points

    def update(i):
        xPred = np.zeros((N + 1, 1)); yPred = np.zeros((N + 1, 1))
        SSpoints_x = np.zeros((numSS_Points, 1)); SSpoints_y = np.zeros((numSS_Points, 1))

        for j in range(0, N + 1):
            xPred[j, 0], yPred[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.PredictedStates[j, 4, i, it],
                                                             LMPCOpenLoopData.PredictedStates[j, 5, i, it])

            if j == 0:
                x = SS_glob[i, 4, it]
                y = SS_glob[i, 5, it]
                psi = SS_glob[i, 3, it]
                l = 0.4;w = 0.2
                car_x = [x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
                         x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
                car_y = [y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
                         y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]

        for j in range(0, numSS_Points):
            SSpoints_x[j, 0], SSpoints_y[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.SSused[4, j, i, it],
                                                                       LMPCOpenLoopData.SSused[5, j, i, it])
        SSpoints.set_data(SSpoints_x, SSpoints_y)

        line.set_data(xPred, yPred)

        rec.set_xy(np.array([car_x, car_y]).T)

    anim = FuncAnimation(fig, update, frames=np.arange(0, int(LMPController.TimeSS[it])), interval=100)

    anim.save('ClosedLoop.gif', dpi=80, writer='imagemagick')


def pcol(A, fname='tmp.png', do_outline=0, height=300, width=300):

    """\
    Use a matlab-like 'pcolor' function to display the large elements
    of a matrix using the Python Imaging Library.

    Arguments:
    A          Input Numpy matrix
    fname      Output filename to which to dump the graphics (default 'tmp.png')
    do_outline Whether or not to print an outline around the block (default 0)
    height     The height of the image (default 300)
    width      The width of the image (default 300)

    Example:
    from Numeric import identity,Float
    a = identity(10,Float)
    pcolor_matrix_pil(a)
    """
    #import Pillow.Image, Pillow.ImageDraw
    import PIL.Image
    import PIL.ImageDraw
    img = PIL.Image.new("RGB", (width, height), (255, 255, 255))
    draw = PIL.ImageDraw.Draw(img)

    n, m = A.shape
    if n > width or m > height:
        raise "Rectangle too big %d %d %d %d" % (n, m, width, height)
    for i in range(n):
        xmin = width * i / float(n)
        xmax = width * (i + 1) / float(n)
        for j in range(m):
            ymin = height * j / float(m)
            ymax = height * (j + 1) / float(m)
            color = get_color(A[i, j])
            if do_outline:
                draw.rectangle((xmin, ymin, xmax, ymax), fill=color,
                               outline=(0, 0, 0))
            else:
                draw.rectangle((xmin, ymin, xmax, ymax), fill=color)

    img.save(fname)
    return


def get_color(a, colors=None):
    """\
    Convert a float value to one of a continuous range of colors.
    Rewritten to use recipe 9.10 from the Python Cookbook.
    """
    if colors is None:
        colors = ["lightgrey", "steelblue", "mediumseagreen","firebrick",  "gold", "slateblue", "turquoise"]
    return colors[a]



def Save_statesAnimation(map, LMPCOpenLoopData, LMPController, it):
    SS_glob = LMPController.SS_glob
    TimeSS = LMPController.TimeSS
    SS = LMPController.SS
    uSS = LMPController.uSS

    xdata = []; ydata = []
    fig = plt.figure()
    fig.set_tight_layout(True)

    axvx = fig.add_subplot(3, 2, 1)
    plt.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 0, it], '-ok', label="Closed-loop trajectory")
    lineSSvx, = axvx.plot(xdata, ydata, 'sb-', label="SS")
    linevx, = axvx.plot(xdata, ydata, 'or-', label="Predicted Trajectory")
    plt.ylabel("vx")
    plt.xlabel("s")

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)

    axvy = fig.add_subplot(3, 2, 2)
    axvy.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 1, it], '-ok')
    lineSSvy, = axvy.plot(xdata, ydata, 'sb-')
    linevy, = axvy.plot(xdata, ydata, 'or-')
    plt.ylabel("vy")
    plt.xlabel("s")

    axwz = fig.add_subplot(3, 2, 3)
    axwz.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 2, it], '-ok')
    lineSSwz, = axwz.plot(xdata, ydata, 'sb-')
    linewz, = axwz.plot(xdata, ydata, 'or-')
    plt.ylabel("wz")
    plt.xlabel("s")

    axepsi = fig.add_subplot(3, 2, 4)
    axepsi.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 3, it], '-ok')
    lineSSepsi, = axepsi.plot(xdata, ydata, 'sb-')
    lineepsi, = axepsi.plot(xdata, ydata, 'or-')
    plt.ylabel("epsi")
    plt.xlabel("s")

    axey = fig.add_subplot(3, 2, 5)
    axey.plot(SS[0:TimeSS[it], 4, it], SS[0:TimeSS[it], 5, it], '-ok')
    lineSSey, = axey.plot(xdata, ydata, 'sb-')
    lineey, = axey.plot(xdata, ydata, 'or-')
    plt.ylabel("ey")
    plt.xlabel("s")

    Points = np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.width)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.width)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    axtr = fig.add_subplot(3, 2, 6)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(SS_glob[0:TimeSS[it], 4, it], SS_glob[0:TimeSS[it], 5, it], '-ok')

    SSpoints_x = []; SSpoints_y = []
    xPred = []; yPred = []
    SSpoints_tr, = axtr.plot(SSpoints_x, SSpoints_y, 'sb')
    line_tr, = axtr.plot(xPred, yPred, '-or')

    N = LMPController.N
    numSS_Points = LMPController.numSS_Points

    def update(i):
        xPred    = LMPCOpenLoopData.PredictedStates[:, :, i, it]
        SSpoints = LMPCOpenLoopData.SSused[:, :, i, it]

        linevx.set_data(xPred[:, 4], xPred[:, 0])
        linevy.set_data(xPred[:, 4], xPred[:, 1])
        linewz.set_data(xPred[:, 4], xPred[:, 2])
        lineepsi.set_data(xPred[:, 4], xPred[:, 3])
        lineey.set_data(xPred[:, 4], xPred[:, 5])

        lineSSvx.set_data(SSpoints[4,:], SSpoints[0,:])
        lineSSvy.set_data(SSpoints[4,:], SSpoints[1,:])
        lineSSwz.set_data(SSpoints[4,:], SSpoints[2,:])
        lineSSepsi.set_data(SSpoints[4,:], SSpoints[3,:])
        lineSSey.set_data(SSpoints[4,:], SSpoints[5,:])

        xPred = np.zeros((N + 1, 1));yPred = np.zeros((N + 1, 1))
        SSpoints_x = np.zeros((numSS_Points, 1));SSpoints_y = np.zeros((numSS_Points, 1))

        for j in range(0, N + 1):
            xPred[j, 0], yPred[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.PredictedStates[j, 4, i, it],
                                                             LMPCOpenLoopData.PredictedStates[j, 5, i, it])

        for j in range(0, numSS_Points):
            SSpoints_x[j, 0], SSpoints_y[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.SSused[4, j, i, it],
                                                                       LMPCOpenLoopData.SSused[5, j, i, it])

        line_tr.set_data(xPred, yPred)
        SSpoints_tr.set_data(SSpoints_x, SSpoints_y)

    anim = FuncAnimation(fig, update, frames=np.arange(0, int(LMPController.TimeSS[it])), interval=100)

    anim.save('ClosedLoopStates.gif', dpi=80, writer='imagemagick')
