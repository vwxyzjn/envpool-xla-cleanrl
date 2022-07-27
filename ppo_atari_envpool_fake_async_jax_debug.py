# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
# https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence

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
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="debug")
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


class RecordLastAction(gym.Wrapper):
    def step(self, action):
        self.last_actions += [action]
        return self.env.step(action)

    def reset(self):
        self.last_actions = []
        return self.env.reset()


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env = RecordLastAction(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class FakeAsyncEnvs:
    def __init__(self, env_id, env_type, num_envs, batch_size, episodic_life, reward_clip, seed):
        self.env_id = env_id
        self.env_type = env_type
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.episodic_life = episodic_life
        self.reward_clip = reward_clip
        self.seed = seed
        self.num_batches = int(num_envs / batch_size)

        self.envs = [
            # envpool.make(
            #     env_id,
            #     env_type="gym",
            #     num_envs=1,
            #     episodic_life=True,
            #     reward_clip=True,
            #     seed=self.seed + i,
            # ) for i in range(num_envs)
            make_env("BreakoutNoFrameskip-v4", self.seed + i, i, False, "")() for i in range(num_envs)
        ]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.original_idxs = np.arange(self.batch_size)
        self.reverse_env_ids = np.zeros(self.batch_size, dtype=np.int32)
        self.env_ids = np.arange(num_envs)
        self.num_dones = 0

    def async_reset(self):
        # self.obs = np.array([env.reset()[0] for env in self.envs]) # envpool specific
        self.obs = np.array([env.reset() for env in self.envs]) # gym specific
        # self.obs = np.array([np.zeros((4, 84, 84)) + i for i in range(self.num_envs)])
        self.rewards = np.zeros(self.num_envs)
        self.dones = np.zeros(self.num_envs)
        if args.debug:
            self.env_idxs = np.random.permutation(self.num_envs)
        else:
            self.env_idxs = np.arange(self.num_envs)
        self.batch_idx = 0

    def send(self, action, env_id):
        for idx, env_idx in enumerate(env_id):
            print("seding action", action[idx], "to env", env_idx)
            # next_obs, reward, done, _ = self.envs[env_idx].step(action[idx:idx+1])  # envpool specific
            next_obs, reward, done, gym_info = self.envs[env_idx].step(action[idx]) # gym specific
            if done: self.num_dones += 1
            self.obs[env_idx] = next_obs
            self.rewards[env_idx] = reward
            self.dones[env_idx] = done
        if self.batch_idx + 1 == self.num_batches:
            if args.debug:
                self.env_idxs = np.random.permutation(self.num_envs)
            else:
                self.env_idxs = np.arange(self.num_envs)
        self.batch_idx = (self.batch_idx + 1) % self.num_batches
        return

    def recv(self):
        batch_idxs = self.env_idxs[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]
        return (
            self.obs[batch_idxs],
            self.rewards[batch_idxs],
            self.dones[batch_idxs],
            {"env_id": self.env_ids[batch_idxs], "terminated": self.dones[batch_idxs]},
        )


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
    envs = FakeAsyncEnvs(
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

    take_idx_fn = jax.vmap(lambda x, idxs: x[idxs]) # TODO: check if this is correct
    @jax.jit
    def compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        env_ids: np.ndarray,
    ):
        env_ids = jnp.asarray(env_ids).reshape(args.num_steps + 1, -1)
        rewards = take_idx_fn(jnp.asarray(rewards).reshape(args.num_steps + 1, -1), env_ids)[1:]  # NOTE: handle rewards off by 1 w/ [1:]
        values = take_idx_fn(jnp.asarray(values).reshape(args.num_steps + 1, -1), env_ids)
        dones = take_idx_fn(jnp.asarray(dones).reshape(args.num_steps + 1, -1), env_ids)

        next_value = values[-1]
        next_done = dones[-1]
        advantages = jnp.zeros((args.num_steps, args.num_envs))
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages = advantages.at[t].set(lastgaelam)
        return advantages, advantages + values[:-1]

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        obs,
        actions,
        logprobs,
        dones,
        values,
        advantages,
        returns,
        env_ids,
        key: jax.random.PRNGKey,
    ):
        env_ids = jnp.asarray(env_ids).reshape(args.num_steps + 1, -1)
        b_obs = take_idx_fn(jnp.asarray(obs).reshape((args.num_steps + 1, -1,)+ envs.single_observation_space.shape), env_ids)[:-1].reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = take_idx_fn(jnp.asarray(logprobs).reshape(args.num_steps + 1, -1), env_ids)[:-1].reshape(-1)
        b_actions = take_idx_fn(jnp.asarray(actions).reshape(args.num_steps + 1, -1), env_ids)[:-1].reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
            newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
            logratio = newlogprob - logp
            ratio = jnp.exp(logratio)
            approx_kl = ((ratio - 1) - logratio).mean()

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

        # clipfracs = []
        for _ in range(args.update_epochs):
            key, subkey = jax.random.split(key)
            b_inds = jax.random.permutation(subkey, args.batch_size, independent=True)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                )
                agent_state = agent_state.apply_gradients(grads=grads)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    async_update = int((args.num_envs / args.async_batch_size))

    # put data in the last index
    sampled_actions = np.random.randint(0, envs.action_space.n, (args.num_steps + 2, args.num_envs))
    print(sampled_actions.sum())
    episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((args.num_envs,), dtype=np.float32)
    obs = []
    dones = []
    actions = []
    logprobs = []
    values = []
    env_ids = []
    rewards = []
    envs.async_reset()
    o = 0
    g = 0
    for _ in range(async_update):
        next_obs, next_reward, next_done, info = envs.recv()
        global_step += len(next_done)
        env_id = info["env_id"]
        action, logprob, value, key = get_action_and_value(agent_state, next_obs, key)
        # envs.send(np.array(action), env_id)
        envs.send(sampled_actions[g][env_id], env_id)
        actions.append(sampled_actions[g][env_id])
        o += 1
        if o % async_update == 0:
            g += 1
        obs.append(next_obs)
        dones.append(next_done)
        values.append(value)
        # actions.append(action)
        logprobs.append(logprob)
        env_ids.append(env_id)
        rewards.append(next_reward)
    # raise
    for update in range(1, args.num_updates + 2):
        # roll over data from last rollout phase
        obs = obs[-async_update:]
        dones = dones[-async_update:]
        actions = actions[-async_update:]
        logprobs = logprobs[-async_update:]
        values = values[-async_update:]
        env_ids = env_ids[-async_update:]
        rewards = rewards[-async_update:] # TODO: fix rewards off by one
        for step in range(async_update, (args.num_steps + 1) * async_update): # num_steps + 1 to get the states for value bootstrapping.
            next_obs, next_reward, next_done, info = envs.recv()
            global_step += len(next_done)
            env_id = info["env_id"]
            action, logprob, value, _ = get_action_and_value(agent_state, next_obs, key) # TODO: fix key
            # envs.send(np.array(action), env_id)
            envs.send(sampled_actions[g][env_id], env_id)
            actions.append(sampled_actions[g][env_id])
            o += 1
            if o % async_update == 0:
                g += 1
            obs.append(next_obs)
            dones.append(next_done)
            values.append(value)
            # actions.append(action)
            logprobs.append(logprob)
            env_ids.append(env_id)
            rewards.append(next_reward)
            episode_returns[env_id] += next_reward
            returned_episode_returns[env_id] = np.where(info["terminated"], episode_returns[env_id], returned_episode_returns[env_id])
            episode_returns[env_id] *= (1 - info["terminated"])
            # print(f"step - {step} envs.envs[0].last_action", envs.envs[0].last_action)
            # if np.array(rewards).sum() > 0:
            #     raise
        
        avg_episodic_return = np.mean(returned_episode_returns)
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        # NOTE!!!!!!!!!!!!! what is env_ids for the rewards?
        advantages, returns = compute_gae(rewards, values, dones, env_ids)
        print(key)
        print("actions.sum()", np.array(actions).sum())
        print("rewards.sum()", np.array(rewards).sum())
        print("returns.sum()", np.array(returns).sum())
        print("dones.sum()", np.array(dones).sum())
        print("envs.envs[0].last_actions", envs.envs[0].last_actions)
        raise
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, _ = update_ppo(  # TODO: fix key
            agent_state,
            obs,
            actions,
            logprobs,
            dones,
            values,
            advantages,
            returns,
            env_ids,
            key,
        )

        print("v_loss", v_loss)
        if update == 1:
            raise
        # raise
        # print("network_params", agent_state.params.network_params["params"]["Dense_0"]["kernel"].sum())
        # print("actor_params", agent_state.params.actor_params["params"]["Dense_0"]["kernel"].sum())
        # print("critic_params", agent_state.params.critic_params["params"]["Dense_0"]["kernel"].sum())
        writer.add_scalar("stats/advantages", advantages.mean().item(), global_step)
        writer.add_scalar("stats/returns", returns.mean().item(), global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
