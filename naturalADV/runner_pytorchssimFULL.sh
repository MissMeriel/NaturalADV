#!/usr/bin/bash

#SBATCH --mem-per-cpu=60Gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ms7nk@virginia.edu

. .venv-nat/bin/activate

# python3 generate_similar_natural_patches.py

# if [[ -z "$SLURM_JOB_ID" ]]; then
#    SLURM_JOB_ID="TESTRUN-MSELOSS"
# fi

timestamp="$(date +%Y%m%d)"
iters=100
# # # weights=(0 0.001 0.01 0.1 0.25 0.5 0.75 0.9 0.999 1)
# # # weights=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
weights=(0.1) # 0.25 0.5)
# stable diffusion starting billboard
patch="CLEAR"
# patches=("CLEAR" "MONDRIAN" "ROUSSEAU" "GTA" "RACING" "MUSEUM")
pertsizes=(10)
resultsparentdir="./meriels-test-results/results-topo1left-ssimwindow3-iters"$iters"-pertsize"$pertsize"-PYTORCHSSIM9-PAIR-nonoise-$timestamp-$SLURM_JOB_ID/"
perturbations=($(ls -dt /p/sdbb/natadv/highstrength-experiment1/high-strength-billboard-library/*.png))
# perturbations=($(ls -dt /p/sdbb/natadv/highstrength-experiment2/scenario2-billboards/*))
for pertsize in "${pertsizes[@]}"; do
    # for origpertfile in "${perturbations[@]}"; do
    origpertfile="/p/sdbb/natadv/highstrength-experiment1/high-strength-billboard-library/21-sdbb-5-25-400-cuton24-rs0.6-inputdivFalse-3_2-20_36-WE7401.png"
    for w in "${weights[@]}"; do
        filename=$(basename -- "$origpertfile")
        # extension="${origpertfile##*.}"
        filename="${filename%.*}"
        resultsdir="$resultsparentdir/$filename/"
        a=$(bc <<< "scale=2; 1 - $w")
        echo RUNNING test_metrics_pytorchssimFULL.py with args $resultsdir $patch ssim weight $w pred weight $a iters $iters
        python3 test_metrics_pytorchssimFULL.py --resultsparentdir $resultsdir --origpertfile $origpertfile --diffusionimage $patch --pertsize $pertsize --ssimweight $w --predweight $a --iters $iters --usedmpert --useunpertseq #--addnoise --weighted 
    done
    # done
done