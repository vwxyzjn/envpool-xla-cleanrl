# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence
from functools import partial

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import envpool
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Breakout-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--async-batch-size", type=int, default=4,
        help="the envpool's batch size in the async mode")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    # fmt: on
    return args


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        batch_size=args.async_batch_size,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params,
            actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
            critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()]))),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    # modifed from https://github.com/sail-sg/envpool/blob/1eedd34902507011dc17f8ab6ca956a3d47431d7/examples/ppo_atari/gae.py#L23
    # @jax.jit
    def compute_gae(
        env_ids: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ):
        dones = jnp.asarray(dones)
        values = jnp.asarray(values)
        env_ids = jnp.asarray(env_ids)
        rewards = jnp.asarray(rewards)

        T, B = env_ids.shape
        final_env_id_checked = jnp.zeros(args.num_envs, jnp.int32) - 1
        final_env_ids = jnp.zeros_like(dones, jnp.int32)
        advantages = jnp.zeros((T, B))
        lastgaelam = jnp.zeros(args.num_envs)
        lastdones = jnp.zeros(args.num_envs) + 1
        lastvalues = jnp.zeros(args.num_envs)

        # TODO: the rewards have different shape — is this going to be an issue?
        for t in reversed(range(T)):
            eid = env_ids[t]
            nextnonterminal = 1.0 - lastdones[eid]
            nextvalues = lastvalues[eid]
            delta = jnp.where(final_env_id_checked[eid] == -1, 0, rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t])
            advantages = advantages.at[t].set(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam[eid])
            final_env_ids = final_env_ids.at[t].set(jnp.where(final_env_id_checked[eid] == 1, 1, 0))
            final_env_id_checked = final_env_id_checked.at[eid].set(jnp.where(final_env_id_checked[eid] == -1, 1, final_env_id_checked[eid]))

            # the last_ variables keeps track of the actual `num_steps`
            lastgaelam = lastgaelam.at[eid].set(advantages[t])
            lastdones = lastdones.at[eid].set(dones[t])
            lastvalues = lastvalues.at[eid].set(values[t])
        return advantages, advantages + values, final_env_id_checked, final_env_ids

    def compute_gae_once(carry, inp):
        lastvalues, lastdones, advantages, lastgaelam, final_env_ids, final_env_id_checked = carry
        done, value, eid, reward, = inp
        nextnonterminal = 1.0 - lastdones[eid]
        nextvalues = lastvalues[eid]
        delta = jnp.where(final_env_id_checked[eid] == -1, 0, reward + args.gamma * nextvalues * nextnonterminal - value)
        advantages = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam[eid]
        final_env_ids = jnp.where(final_env_id_checked[eid] == 1, 1, 0)
        final_env_id_checked = final_env_id_checked.at[eid].set(jnp.where(final_env_id_checked[eid] == -1, 1, final_env_id_checked[eid]))

        # the last_ variables keeps track of the actual `num_steps`
        lastgaelam = lastgaelam.at[eid].set(advantages)
        lastdones = lastdones.at[eid].set(done)
        lastvalues = lastvalues.at[eid].set(value)
        return (lastvalues, lastdones, advantages, lastgaelam, final_env_ids, final_env_id_checked), (advantages, final_env_ids)


    # @jax.jit
    def compute_gae_new(
        env_ids: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ):
        dones = jnp.asarray(dones)
        values = jnp.asarray(values)
        env_ids = jnp.asarray(env_ids)
        rewards = jnp.asarray(rewards)

        T, B = env_ids.shape
        final_env_id_checked = jnp.zeros(args.num_envs, jnp.int32) - 1
        # final_env_ids = jnp.zeros_like(dones, jnp.int32)
        final_env_ids = jnp.zeros(B, jnp.int32)
        # advantages = jnp.zeros((T, B))
        advantages = jnp.zeros(B)
        lastgaelam = jnp.zeros(args.num_envs)
        lastdones = jnp.zeros(args.num_envs) + 1
        lastvalues = jnp.zeros(args.num_envs)

        (_, _, _, _, final_env_ids, final_env_id_checked), (advantages, final_env_ids) = jax.lax.scan(
            compute_gae_once, (
                lastvalues,
                lastdones,
                advantages,
                lastgaelam,
                final_env_ids,
                final_env_id_checked,
            ), (
                dones,
                values,
                env_ids,
                rewards,
            ), reverse=True
        )
        return advantages, advantages + values, final_env_id_checked, final_env_ids


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    async_update = int((args.num_envs / args.async_batch_size))

    # put data in the last index
    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((args.num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.num_envs,), dtype=np.float32)
    obs = []
    dones = []
    actions = []
    logprobs = []
    values = []
    env_ids = []
    rewards = []
    envs.async_reset()
    final_env_ids = np.zeros((async_update, args.async_batch_size), dtype=np.int32) # TODO: issue: there is no guarantee that the first two batch contain all the envs
    for _ in range(async_update):
        next_obs, next_reward, next_done, info = envs.recv()
        global_step += len(next_done)
        env_id = info["env_id"]
        action, logprob, value, key = get_action_and_value(agent_state, next_obs, key)
        envs.send(np.array(action), env_id)
        obs.append(next_obs)
        dones.append(next_done)
        values.append(value)
        actions.append(action)
        logprobs.append(logprob)
        env_ids.append(env_id)
        rewards.append(next_reward)

    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        obs = []
        dones = []
        actions = []
        logprobs = []
        values = []
        env_ids = []
        rewards = []
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        env_send_time = 0

        # NOTE: This is a major difference from the sync version:
        # at the end of the rollout phase, the sync version will have the next observation
        # ready for the value bootstrap, but the async version will not have it.
        # for this reason we do `num_steps + 1`` to get the extra states for value bootstrapping.
        # but note that the extra states are not used for the loss computation in the next iteration,
        # while the sync version will use the extra state for the loss computation.
        for step in range(async_update, (args.num_steps + 1) * async_update): # num_steps + 1 to get the states for value bootstrapping.
            env_recv_time_start = time.time()
            next_obs, next_reward, next_done, info = envs.recv()
            env_recv_time += time.time() - env_recv_time_start
            global_step += len(next_done)
            env_id = info["env_id"]

            inference_time_start = time.time()
            action, logprob, value, key = get_action_and_value(agent_state, next_obs, key)
            inference_time += time.time() - inference_time_start

            env_send_time_start = time.time()
            envs.send(np.array(action), env_id)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()
            obs.append(next_obs)
            dones.append(next_done)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)
            env_ids.append(env_id)
            rewards.append(next_reward)
            episode_returns[env_id] += info["reward"]
            returned_episode_returns[env_id] = np.where(info["terminated"], episode_returns[env_id], returned_episode_returns[env_id])
            episode_returns[env_id] *= (1 - info["terminated"])
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(info["terminated"], episode_lengths[env_id], returned_episode_lengths[env_id])
            episode_lengths[env_id] *= (1 - info["terminated"])
            storage_time += time.time() - storage_time_start

        avg_episodic_return = np.mean(returned_episode_returns)
        # print(returned_episode_returns)
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)

        # count env steps
        # np.unique(np.asarray(env_ids).reshape(args.num_steps, -1), return_counts=True)
        
        advantages_old, returns_old, final_env_id_checked_old, final_env_ids_old = compute_gae(env_ids, rewards, values, dones)
        advantages, returns, final_env_id_checked, final_env_ids = compute_gae_new(env_ids, rewards, values, dones)
        import chex
        chex.assert_trees_all_close(advantages_old, advantages)
        chex.assert_trees_all_close(returns_old, returns)
        chex.assert_trees_all_close(final_env_id_checked_old, final_env_id_checked)
        chex.assert_trees_all_close(final_env_ids_old, final_env_ids)
        print(advantages.sum(), returns.sum(), final_env_id_checked.sum(), final_env_ids.sum())
        break

    envs.close()
    writer.close()
