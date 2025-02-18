#!/usr/bin/bash

#SBATCH --mem-per-cpu=28Gb

. .venv-nat/bin/activate

if [[ -z "$SLURM_JOB_ID" ]]; then
   SLURM_JOB_ID="TESTRUN-MSELOSS"
fi

timestamp="$(date +%Y%m%d)"
iters=50
# squeeze
net=alex 
resultsparentdir="./results-LPIPS4-USINGFORWARD-USINGFULLIMGS-$net-$timestamp-$SLURM_JOB_ID/"
# weights=(0 0.001 0.01 0.1 0.25 0.5 0.75 0.9 0.999 1)
weights=(0 0.1 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.9 1)

# stable diffusion starting billboard
patches=("CLEAR") # "MONDRIAN" "ROUSSEAU" "GTA" "RACING" "MUSEUM")
for patch in "${patches[@]}"; do
    for w in "${weights[@]}"; do
        a=$(bc <<< "scale=2; 1 - $w")
        echo RUNNING test_metrics_lpips4.py with args $resultsparentdir $patch ssim weight $w pred weight $a iters $iters
        python3 test_metrics_lpips4.py --resultsparentdir $resultsparentdir --net $net --diffusionimage $patch --ssimweight $w --predweight $a --iters $iters --useunpertseq #--weighted 
    done
done