import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, ampl, center, width):
    return ampl*np.exp(-(x-center)**2/width)


def make_dist(x, ampl, pos, std):
    for i in range(len(ampl_1)):
        peak = gaussian(x, ampl[i], pos[i], std[i])
        if "y" in locals():
            y += peak
        else:
            y = peak.copy()
        
    return y


# PARAMETERS
x = np.linspace(-5, 60, 1001)

ampl_1 = np.array([2, 1.4, 1., 1.2, 1, 0.8])
pos_1 = np.array([3, 7, 11, 15, 19, 23])
std_1 = np.array([4, 3, 4, 4, 3, 4])
vert_shift_1 = 0.02

ampl_2 = np.array([2.0, 1.0, 1.5, 1.0, 0.9, 0.7])
pos_2 = np.array([27, 31, 35, 39, 43, 47])
std_2 = np.array([4, 3, 4, 4, 3, 4])
vert_shift_2 = 0.02


# PRODUCE CURVES
y_1 = make_dist(x, ampl_1, pos_1, std_1) + vert_shift_1
#y_2 = make_dist(x, ampl_1, pos_1+24, std_1) + vert_shift_1
y_2 = make_dist(x, ampl_2, pos_2, std_2) + vert_shift_2


# PLOT CURVES
fig = plt.figure(figsize=(20,5), dpi=80)
ax = fig.add_subplot(1, 1, 1)
baseline = np.zeros(len(x))
#ax.plot(x, y_2, '-', color = 'royalblue', linewidth=3)
#ax.plot(x, y_1, '-', color = 'dodgerblue', linewidth=3)
ax.fill_between(x, baseline, y_2, where=baseline<y_2, alpha=0.5, facecolor="darkblue", edgecolor="none")
ax.fill_between(x, baseline, y_1, where=baseline<y_1, alpha=0.5, facecolor="darkslategray", edgecolor="none")

# DRAW ARROWS
ax.arrow(x[0], 0, x[-1]-x[0], 0, head_width=0.1, head_length=3, fc='k', ec='k')
ax.arrow(x[0], 0, 0, np.max(y_2)-0.3, head_width=1, head_length=0.3, fc='k', ec='k')

# CLEAN UP EVERYTHING
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xlim([x[0]-2, x[-1]+3])
ax.set_ylim([-0.2, np.max(y_1)])
plt.axis("off")
plt.savefig("logo_DynDen_curves.svg")
plt.show()