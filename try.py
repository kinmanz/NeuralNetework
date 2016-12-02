import matplotlib.pyplot as plt
import numpy as np


#   set up figure for plotting:
fig = plt.figure()
ax = fig.add_subplot(111)

#    plot limits
ax.set_xlim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))
ax.set_ylim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))

#    colors
colors = ['b', 'g', 'c']

for i_t in range(0, np.shape(t)[1]):
    ax.clear()
    plt.hold(True)
    #    plot limits
    ax.set_xlim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))
    ax.set_ylim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))
    for i in range(0, N):
        ax.plot(x_matrix[i_t, i], y_matrix[i_t, i], 's', color=colors[0, i])

    plt.pause(0.0001)