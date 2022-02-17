#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=12                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=124G                        # Ask for 32 GB of RAM
#SBATCH --time=6:00:00                   # The job will run for 3 hours

#SBATCH -o /scratch/<user>/slurm-%j.out  # Write the log in $SCRATCH

#--time=1:0:0 --account=rrg-bengioy-ad --gres=gpu:1 --cpus-per-task=12 --mem=124G

module load python/3.7
cd
source ift6163/bin/activate
cd scratch/ift6163_homeworks/hw2/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin


python run_hw2_mb.py exp_name=q1_cheetah_n500_arch1x32 \
                    env_name=cheetah-ift6163-v0 \
                    num_agent_train_steps_per_iter=500 \
                    n_layers=1 \
                    size=32

python run_hw2_mb.py exp_name=q1_cheetah_n5_arch2x250 \
                      env_name=cheetah-ift6163-v0 \
                      num_agent_train_steps_per_iter=5 \
                      n_layers=2 \
                      size=250

python run_hw2_mb.py exp_name=q1_cheetah_n500_arch2x250 \
                      env_name=cheetah-ift6163-v0 \
                      num_agent_train_steps_per_iter=500 \
                      n_layers=2 \
                      size=250