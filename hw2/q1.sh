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