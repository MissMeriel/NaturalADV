#!/usr/bin/bash

#SBATCH --mem-per-cpu=24Gb

. .venv-nat/bin/activate

if [[ -z "$SLURM_JOB_ID" ]]; then
   SLURM_JOB_ID="TESTRUN-MSELOSS"
fi

timestamp="$(date +%s)"
iters=1000
resultsparentdir="/results-KORNIAWELSCH-NONORM-PREDMSELOSS-iters$iters-$timestamp-$SLURM_JOB_ID/"
weights=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)



# clear starting billboard
for w in "${weights[@]}"; do
    # for a in "${weights[@]}"; do
    a=$(bc <<< "scale=2; 1 - $w")
    echo RUNNING test_metrics_dm2clear_welsch.py with args $resultsparentdir criterion weight $w pred weight $a iters $iters
    python3 test_metrics_dm2clear_welsch.py --resultsparentdir $resultsparentdir --criterionweight $w --predweight $a --iters $iters
done


# stable diffusion starting billboard
patches=("MONDRIAN" "ROUSSEAU" "GTA" "RACING" "MUSEUM")
for patch in "${patches[@]}"; do
    for w in "${weights[@]}"; do
        a=$(bc <<< "scale=2; 1 - $w")
        echo RUNNING test_metrics_dm2clear_welsch.py with args $resultsparentdir $patch criterion weight $w pred weight $a iters $iters
        python3 test_metrics_dm2clear_welsch.py --resultsparentdir $resultsparentdir --diffusionimage $patch --criterionweight $w --predweight $a --iters $iters
    done
done

