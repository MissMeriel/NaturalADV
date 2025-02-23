import os
from pathlib import Path
import pickle
import numpy as np

parentdir = "C:/Users/Meriel/Documents/GitHub/superdeepbillboard/simulation/results/dissertation-topo1left-cleanrerun-PA3XG3/"
parentdir = "C:/Users/Meriel/Documents/GitHub/superdeepbillboard/simulation/results/dissertation-topo1left-50testruns-KQ0JET/"
dirs = [i for i in os.listdir(parentdir) if "results-sdbb-15-5-400-cuton32-rs0.6-inputdivFalse-" in i]
# print(dirs)
success_rates = [] #np.zeros(len(dirs))
for j, d in enumerate(dirs):
    print(d)
    # print(os.listdir(parentdir + d))
    try:
        with open(f"{parentdir}{d}/results.pickle", "rb") as f:
            hm = pickle.load(f)
            # print(hm.keys())
            # dict_keys(['testruns_deviation', 'testruns_dists', 'testruns_ys', 'testruns_mse', 'testruns_error', 'testruns_errors', 'testruns_outcomes', 'time_to_run_technique', 'unperturbed_dists', 'unperturbed_deviation', 'unperturbed_traj', 'unperturbed_all_ys', 'pertrun_all_ys', 'pertrun_outcome', 'pertrun_traj', 'pertrun_deviation', 'pertrun_dist', 'num_billboards', 'MAE_collection_sequence', 'testruns_trajs', 'testruns_all_ys', 'all_trajs', 'modelname', 'technique', 'direction', 'lossname', 'bbsize', 'its', 'nl'])
            success_rate = sum([1 for i in hm['testruns_outcomes'] if ("D" in i or "LT" in i)]) / len(hm['testruns_outcomes'])
            print(f"{success_rate=:.3f} ({len(hm['testruns_outcomes'])} runs)")
            # success_rates[j] = success_rate
            success_rates.append(success_rate)
    except FileNotFoundError as e:
        print(e)
print(f"Success rate out of {len(success_rates)} billboards ({len(hm['testruns_outcomes'])} test runs each): mean={np.mean(success_rates):.3f} std={np.std(success_rates):.3f}")
print(sorted(success_rates))
