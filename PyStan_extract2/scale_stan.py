import datetime
import gc
import os
import pickle
import timeit
import pystan

if __name__ == '__main__':
    start = datetime.datetime.now()
    print("Starting:", start)

    model_code = """
    data {
        int N;
    }
    parameters {
        vector[N] y;
    }
    model {
        y ~ normal(0, 1);
    }
    """
    stan_model = pystan.StanModel(model_code=model_code)

    stan_data = {'N' : 10}

    # REFERENCE TESTING
    ref_fit = stan_model.sampling(data=stan_data, seed=131, check_hmc_diagnostics=False)
    ref_permuted = ref_fit.extract()
    ref_not_permuted = ref_fit.extract(permuted=False, pars=ref_fit.model_pars)
    ref_array = ref_fit.extract(permuted=False)

    path = f"reference_results_{pystan.__version__}" + "{}.pickle"
    i = 0
    while os.path.exists(path.format(i)):
        i += 1

    with open(path.format(i), "wb") as f:
        pickle.dump([ref_fit, ref_permuted, ref_not_permuted, ref_array], f)

    # TIMING

    timing_dict = {}
    for n in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
        for s in [100, 500, 1000, 1500, 2000]:
            k = 1000 if n < 1000 else 100 if n < 10000 else 10
            print(n, s, k)
            stan_data = {'N' : n}
            fit = stan_model.sampling(data=stan_data, iter=s, seed=131, check_hmc_diagnostics=False)

            t_permuted = timeit.timeit("fit.extract(permuted=True)", 'gc.enable(); from __main__ import fit', number=k)
            t_not_permuted = timeit.timeit("fit.extract(permuted=False, pars=fit.model_pars)", 'gc.enable(); from __main__ import fit', number=k)
            t_array = timeit.timeit("fit.extract(permuted=False)", 'gc.enable(); from __main__ import fit', number=k)

            timing_dict[(n, s)] = [t_permuted, t_not_permuted, t_array, k]
            [print(s, " ::: ", item) for s,item in zip(['permuted', 'not_permuted', 'array'],timing_dict[(n,s)])]
            gc.collect()

    with open(path.format("timing_{}".format(i)), "wb") as f:
        pickle.dump(timing_dict, f)

    end = datetime.datetime.now()
    print("Ending:", end)
    print("Duration: ", end-start)
    print("Done")
