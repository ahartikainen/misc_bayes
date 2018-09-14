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

random_state = np.random.RandomState(seed=131)

print("Starting to sample", flush=True)
neffs = {}
for dist_name, (scipy_dist, stan_dist, parameters) in distributions.items():
    neffs[dist_name] = {}
    print(dist_name, flush=True)
    for params in parameters:
        # unpack parameters
        if dist_name == 'student_t':
            df, location, scale = params
            dist_key = f"{dist_name}_loc_{location}_scale_{scale}_df_{df}"
            stan_filename = f"./stan_model_{dist_name}_loc_{location}_scale_{scale}_df_{df}.pickle.gz"
            # create distribution objects
            scipy_distribution = scipy_dist(loc=location, scale=scale, df=df)
            if os.path.exists(stan_filename):
                with open(stan_filename, "rb") as f:
                    stan_distribution = pickle.loads(f.read(-1))
            else:
                stan_distribution = pystan.StanModel(model_code=stan_dist.format(df, location, scale))
                with open(stan_filename, "wb") as f:
                    f.write(pickle.dumps(stan_distribution))
        else:
            location, scale = params
            dist_key = f"{dist_name}_loc_{location}_scale_{scale}"
            stan_filename = f"./stan_model_{dist_name}_loc_{location}_scale_{scale}.pickle.gz"
            # create distribution objects
            scipy_distribution = scipy_dist(loc=location, scale=scale)
            if os.path.exists(stan_filename):
                with open(stan_filename, "rb") as f:
                    stan_distribution = pickle.loads(f.read(-1))
            else:
                stan_distribution = pystan.StanModel(model_code=stan_dist.format(location, scale))
                with open(stan_filename, "wb") as f:
                    f.write(pickle.dumps(stan_distribution))
        print("dist key: ", dist_key, flush=True)
        scipy_neffs = []
        stan_neffs = []
        for _ in range(10000):
            state = random_state.randint(0, 10000)
            scipy_random_array = scipy_distribution.rvs(size=(4,1000), random_state=state)
            fit = stan_distribution.sampling(algorithm='Fixed_param', iter=1000, warmup=0)
            stan_random_array = az.convert_to_dataset(fit)
            scipy_neff = float(az.stats.effective_n(scipy_random_array))
            stan_neff = float(az.stats.effective_n(stan_random_array, var_names='z').values)
            scipy_neffs.append(scipy_neff)
            stan_neffs.append(stan_neff)
        neffs[dist_name][dist_key] = np.array(scipy_neffs), np.array(stan_neffs)

print("Saving samples", flush=True)
with gzip.open("./neff_samples.pickle.gz", "wb") as f:
    f.write(pickle.dumps(neffs))

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
