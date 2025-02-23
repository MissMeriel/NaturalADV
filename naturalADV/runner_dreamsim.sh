#!/usr/bin/bash

#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

. .venv-nat/bin/activate

if [[ -z "$SLURM_JOB_ID" ]]; then
   SLURM_JOB_ID="TESTRUN-MSELOSS"
fi

timestamp="$(date +%s)"
iters=10000
# resultsparentdir="/results-KORNIA-NEGSSIMLOSS-NONORM-MSELOSS-iters$iters-$timestamp-$SLURM_JOB_ID/"
resultsparentdir="/results-dreamsim-TEST/$timestamp-$SLURM_JOB_ID/"
# weights=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
weights=(1.0)



# clear starting billboard
for w in "${weights[@]}"; do
    # for a in "${weights[@]}"; do
    a=$(bc <<< "scale=2; 1 - $w")
    echo RUNNING test_metrics_dm2clear_dreamsim.py with args $resultsparentdir ssim weight $w pred weight $a iters $iters
    python3 test_metrics_dm2clear_dreamsim.py --resultsparentdir $resultsparentdir  --ssimweight $w --predweight $a --iters $iters
done


# stable diffusion starting billboard
# patches=("MONDRIAN" "ROUSSEAU" "GTA" "RACING" "MUSEUM")
# for patch in "${patches[@]}"; do
#     for w in "${weights[@]}"; do
#         a=$(bc <<< "scale=2; 1 - $w")
#         echo RUNNING test_metrics_dm2clear_kornia.py with args $resultsparentdir $patch ssim weight $w pred weight $a iters $iters
#         python3 test_metrics_dm2clear_kornia.py --resultsparentdir $resultsparentdir --diffusionimage $patch --ssimweight $w --predweight $a --iters $iters
#     done
# done

