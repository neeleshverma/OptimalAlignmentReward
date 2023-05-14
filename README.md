# Optimal Alignment Reward
The underlying algorithm is IQL. The code for IQL is taken from https://github.com/ikostrikov/implicit_q_learning .  

The code can be simply run as -

```
python train_offline.py --workdir /tmp/ot \
        --config configs/iql_mujoco.py \
        --config.expert_dataset_name='hopper-medium-v2' \
        --config.k=1 \
        --config.offline_dataset_name='hopper-medium-v2' \
        --config.use_dataset_reward=True
```

Here, k is number of expert trajectories.  

For solving the Optimal Transport problem, JAX implementation is used https://github.com/ott-jax/ott

Contact Info : neverma@cs.stonybrook.edu
