# envpool-xla-cleanrl (Experimental)

This is an experimental support of leveraging envpool's XLA interface with CleanRL's PPO. To get started, run the following command.

```
poetry install
poetry run pip install --upgrade "jax[cuda]==0.3.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry run python ppo_atari_envpool_xla_jax.py
```