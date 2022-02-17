module load python/3.7
cd
source ift6163/bin/activate
cd scratch/ift6163_homeworks/hw2/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jgsimard/.mujoco/mujoco200/bin


#python run_hw2_mb.py exp_name=q3_obstacles \
#                      env_name=obstacles-ift6163-v0 \
#                      num_agent_train_steps_per_iter=20 \
#                      batch_size_initial=5000 \
#                      batch_size=1000 \
#                      mpc_horizon=10 \
#                      n_iter=12 \
#                      video_log_freq=-1


#python run_hw2_mb.py exp_name=q3_reacher \
#                      env_name=reacher-ift6163-v0 \
#                      mpc_horizon=10 \
#                      num_agent_train_steps_per_iter=1000 \
#                      batch_size_initial=5000 \
#                      batch_size=5000 \
#                      n_iter=15 \
#                      video_log_freq=-1
#
python run_hw2_mb.py exp_name=q3_cheetah \
                      env_name=cheetah-ift6163-v0 \
                      mpc_horizon=15 \
                      num_agent_train_steps_per_iter=1500 \
                      batch_size_initial=5000 \
                      batch_size=5000 \
                      n_iter=20 \
                      video_log_freq=-1