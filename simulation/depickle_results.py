import pickle
import numpy as np
picklefile = "C:/Users/Meriel/Documents/GitHub/contextualvalidation/simulation/results/contextualvalidation-sdpatch0.001ssim-30runs-FE5GOP/results-28-cuton11_1-0_34-CLMMBX/results.pickle"
with open(picklefile, "rb") as f:
    results = pickle.load(f)
    print(results.keys())
all_outcomes = results['testruns_outcomes']
print(f"{all_outcomes=}")
failures = 0
for o in all_outcomes:
    if o == "LT" or "D" in o:
        failures += 1
print(f"Failures={failures} ({(failures / len(all_outcomes) * 100):.1f}%)")
print(f"Results saved to {picklefile}")