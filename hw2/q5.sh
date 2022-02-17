#module load python/3.7
#cd
#source ift6163/bin/activate
#cd scratch/ift6163_homeworks/hw2/
#
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin

python run_hw2_mb.py exp_name=q5_cheetah_random \
                      env_name='cheetah-ift6163-v0' \
                      mpc_horizon=15 \
                      num_agent_train_steps_per_iter=1500 \
                      batch_size_initial=5000 \
                      batch_size=5000 \
                      n_iter=5 \
                      video_log_freq=-1 \
                      mpc_action_sampling_strategy='random'

#python run_hw2_mb.py exp_name=q5_cheetah_cem_2 \
#                      env_name='cheetah-ift6163-v0' \
#                      mpc_horizon=15 \
#                      add_sl_noise=true \
#                      num_agent_train_steps_per_iter=1500 \
#                      batch_size_initial=5000 \
#                      batch_size=5000 \
#                      n_iter=5 \
#                      video_log_freq=-1 \
#                      mpc_action_sampling_strategy='cem' \
#                      cem_iterations=2

#python run_hw2_mb.py exp_name=q5_cheetah_cem_4 \
#                      env_name='cheetah-ift6163-v0' \
#                      mpc_horizon=15 \
#                      add_sl_noise=true \
#                      num_agent_train_steps_per_iter=1500 \
#                      batch_size_initial=5000 \
#                      batch_size=5000 \
#                      n_iter=5 \
#                      video_log_freq=-1 \
#                      mpc_action_sampling_strategy='cem' \
#                      cem_iterations=4