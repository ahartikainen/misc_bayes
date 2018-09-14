# disable visual plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pystan
from scipy.stats.distributions import cauchy, norm, t as student_t
import arviz as az
import pickle
import gzip
from datetime import datetime
import os

stime = datetime.now()
print("Starting: ", stime)

distributions = {
    'cauchy' : (cauchy, "generated quantities {{ real z; z = cauchy_rng({},{});}}", [(0,1)]),
    'student_t' : (student_t, "generated quantities {{ real z; z = student_t_rng({},{},{});}}", [(2,0,1), (3,0,1), (10,0,1)]),
    'normal' : (norm, "generated quantities {{ real z; z = normal_rng({},{});}}", [(0,1)]),
}

print("Reading samples", flush=True)
with gzip.open("./neff_samples.pickle.gz", "rb") as f:
    neffs = pickle.loads(f.read(-1))

print("Starting to plot", flush=True)
for key, eff_ns in neffs.items():
    for key_, (eff_n_scipy, eff_n_stan) in eff_ns.items():
        ax = az.kdeplot(eff_n_scipy, plot_kwargs={'color' : 'k', 'linewidth' : 2}, label=f'scipy', rug=True)
        ax = az.kdeplot(eff_n_stan, plot_kwargs={'color' : 'r', 'ls' : '--', 'linewidth' : 2}, ax=ax, label=f'stan', rug=True)
        ax.axvline(4000, color='k', ls='dotted', ymin=0.1)
        ax.legend(fontsize=20)
        ax.set_yticks([])
        x_ticks = list(map(int, ax.get_xticks()))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=15);
        ax.text(0.02, 0.93, key_.replace("_", " "), transform=ax.transAxes, fontsize=40, horizontalalignment='left', verticalalignment='center')
        fig = ax.figure
        plt.savefig(f"{key_}", dpi=300, bbox_inches='tight')
        plt.close("all")


etime = datetime.now()
duration = etime - stime

print("Finished:", etime)
print("Duration", duration)
