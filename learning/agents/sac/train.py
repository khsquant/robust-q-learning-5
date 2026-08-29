# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import struct
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, NamedTuple

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from agents.sac import checkpoint
from agents.sac import losses as sac_losses
from agents.sac import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.wrapper import Wrapper
import flax
import jax
import jax.numpy as jnp
import optax
from learning.module.wrapper.evaluator import AdvEvaluator, Evaluator
import wandb
import numpy as np
import matplotlib.pyplot as plt

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


class TransitionwithParams(NamedTuple):
  """Transition with the dynamics parameters used for the step."""
  observation: jax.Array
  action: jax.Array
  dynamics_params: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: dict[str, Any]


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  gradient_steps: types.UInt64
  env_steps: types.UInt64
  alpha_optimizer_state: optax.OptState
  alpha_params: Params
  normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _replicate_across_devices(value, local_devices_to_use: int):
  return jax.device_put(
      jax.tree_util.tree_map(
          lambda x: jnp.broadcast_to(
              jnp.asarray(x), (local_devices_to_use,) + jnp.asarray(x).shape
          ),
          value,
      )
  )


def _init_training_state(
    key: PRNGKey,
    obs_size: Union[int, Dict[str, specs.Array]],    
    local_devices_to_use: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_q = jax.random.split(key)
  log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
  alpha_optimizer_state = alpha_optimizer.init(log_alpha)

  policy_params = sac_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = sac_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  normalizer_params = running_statistics.init_state(
    #   specs.Array((obs_size,), jnp.dtype('float32'))
    obs_size
  )

  training_state = TrainingState(
      policy_optimizer_state=policy_optimizer_state,
      policy_params=policy_params,
      q_optimizer_state=q_optimizer_state,
      q_params=q_params,
      target_q_params=q_params,
      gradient_steps=types.UInt64(hi=0, lo=0),
      env_steps=types.UInt64(hi=0, lo=0),
      alpha_optimizer_state=alpha_optimizer_state,
      alpha_params=log_alpha,
      normalizer_params=normalizer_params,
  )
  return _replicate_across_devices(training_state, local_devices_to_use)


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 1024,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    eval_randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    dr_train_ratio = 1.0,
    dr_augmented_critic: bool = False,
):
  """SAC training."""
  process_id = jax.process_index()
  local_devices_to_use = jax.local_device_count()
  if max_devices_per_host is not None:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  device_count = local_devices_to_use * jax.process_count()
  logging.info(
      'local_device_count: %s; total_device_count: %s',
      local_devices_to_use,
      device_count,
  )

  if min_replay_size >= num_timesteps:
    raise ValueError(
        'No training will happen because min_replay_size >= num_timesteps'
    )

  if max_replay_size is None:
    max_replay_size = num_timesteps

  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of run_one_sac_epoch calls per run_sac_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )

  assert num_envs % device_count == 0
  import copy
  env = copy.deepcopy(environment)


  rng = jax.random.PRNGKey(seed)
  rng, key = jax.random.split(rng)
  evaluation_randomization_fn = eval_randomization_fn or randomization_fn
  if hasattr(env,'dr_range') :
    dr_low, dr_high = env.dr_range
    dynamics_param_size = len(dr_low)
    dr_mid = (dr_low + dr_high) / 2.
    dr_scale = (dr_high - dr_low) / 2.
    training_dr_range = (dr_mid - dr_train_ratio*dr_scale, dr_mid + dr_train_ratio*dr_scale)
  else:
    dr_low = dr_high = None
    dynamics_param_size = 0
    training_dr_range = None
  training_randomization_fn = None
  if randomization_fn is not None:
    if training_dr_range is None:
      raise ValueError(
          'SAC domain randomization requires an environment with dr_range.'
      )
    training_randomization_fn = functools.partial(
        randomization_fn,
        dr_range=training_dr_range,
    )
    env = wrap_for_adv_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=training_randomization_fn,
      param_size=len(dr_low),
      dr_range_low=dr_low,
      dr_range_high=dr_high,
    )
  else:
    env = envs.training.wrap(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
    )
  obs_shape = env.observation_size
#   if isinstance(obs_size, Dict):
#     obs_size = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    # raise NotImplementedError('Dictionary observations not implemented in SAC')
  print("SAC OBS SIZE", obs_shape)
  action_size = env.action_size

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  sac_network = network_factory(
      observation_size=obs_shape,
      action_size=action_size,
      param_size=dynamics_param_size,
      preprocess_observations_fn=normalize_fn,
      dr_augmented_critic=dr_augmented_critic,
  )
  make_policy = sac_networks.make_inference_fn(sac_network)

  alpha_optimizer = optax.adam(learning_rate=3e-4)

  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)

  dummy_obs = { key: jnp.zeros(obs_shape[key]) for key in obs_shape } if isinstance(obs_shape, dict) else jnp.zeros((obs_shape,))
  print("dummy_obs", dummy_obs)
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      dynamics_params=jnp.zeros((dynamics_param_size,), dtype=jnp.float32),
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
  )
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
      sac_network=sac_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
      dr_augmented_critic=dr_augmented_critic,
  )
  alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: TransitionwithParams
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry

    key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

    alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
        training_state.alpha_params,
        training_state.policy_params,
        training_state.normalizer_params,
        transitions,
        key_alpha,
        optimizer_state=training_state.alpha_optimizer_state,
    )
    alpha = jnp.exp(training_state.alpha_params)
    (critic_loss, (current_q, next_v)), q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        alpha,
        transitions,
        key_critic,
        optimizer_state=training_state.q_optimizer_state,
    )
    actor_loss, policy_params, policy_optimizer_state = actor_update(
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.q_params,
        alpha,
        transitions,
        key_actor,
        optimizer_state=training_state.policy_optimizer_state,
    )

    new_target_q_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_q_params,
        q_params,
    )

    metrics = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'alpha': jnp.exp(alpha_params),
        'current_q_min' : current_q.min(),
        'current_q_max' : current_q.max(),
        'current_q_mean' : current_q.mean(),
        'next_v_min' : next_v.min(),
        'next_v_max' : next_v.max(),
        'next_v_mean' : next_v.mean(),
    }

    new_training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        normalizer_params=training_state.normalizer_params,
    )
    return (new_training_state, key), metrics
  def adv_step(
      env,
      env_state,
      policy,
      key: PRNGKey,
      extra_fields: Sequence[str] = (),
    ):
      step_key, key = jax.random.split(key)
      actions, policy_extras = policy(env_state.obs, key)
      # dynamics_params = jax.random.uniform(key=step_key, shape=(num_envs,len(dr_low)), minval=dr_low, maxval=dr_high)
      if randomization_fn is not None:
        dynamics_params = jax.random.uniform(
          key=step_key,
          shape=env_state.info["dr_params"].shape,
          minval=dr_low,
          maxval=dr_high,
        )
        params = (
          env_state.info["dr_params"] * (1 - env_state.done[..., None])
          + dynamics_params * env_state.done[..., None]
        )
        nstate = env.step(env_state, actions, params)
      else:
        params = jnp.zeros(
            env_state.reward.shape + (dynamics_param_size,), dtype=jnp.float32
        )
        nstate = env.step(env_state, actions)
      state_extras = {x: nstate.info[x] for x in extra_fields}
      return nstate, TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
          observation=env_state.obs,
          action=actions,
          dynamics_params=params,
          reward=nstate.reward,
          discount=1 - nstate.done,
          next_observation= nstate.obs,
          extras={'policy_extras': policy_extras, 'state_extras': state_extras},)
  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      envs.State,
      ReplayBufferState,
  ]:
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = adv_step(
        env, env_state, policy, key, extra_fields=('truncation',)
    )
    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    simul_info ={
      "simul/reward_mean" : transitions.reward.mean(),
      "simul/reward_std" : transitions.reward.std(),
      "simul/reward_max" : transitions.reward.max(),
      "simul/reward_min" : transitions.reward.min(),
      # "simul/params_mean" : transitions.dynamics_params.mean(),
      # "simul/params_std" : transitions.dynamics_params.std(),

    }
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, env_state, buffer_state, simul_info

  def training_step(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      TrainingState,
      envs.State,
      ReplayBufferState,
      Metrics,
  ]:
    experience_key, training_key = jax.random.split(key)
    normalizer_params, env_state, buffer_state, simul_info = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        env_state,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
    )

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    # Change the front dimension of transitions so 'update_step' is called
    # grad_updates_per_step times by the scan.
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )

    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
    metrics.update(simul_info)
    return training_state, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state, simul_info = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          env_state,
          buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), ()

    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_prefill_actor_steps,
    )[0]

  prefill_replay_buffer = jax.pmap(
      prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME
  )

  def training_epoch(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

    def f(carry, unused_t):
      ts, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, metrics = training_step(ts, es, bs, k)
      return (ts, es, bs, new_key), metrics

    (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, env_state, buffer_state, metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, env_state, buffer_state, metrics) = training_epoch(
        training_state, env_state, buffer_state, key
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        env_steps_per_actor_step * num_training_steps_per_epoch
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  global_key, local_key = jax.random.split(rng)
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )
  env_state = jax.pmap(env.reset)(env_keys)

  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  print("SAC OBS SHAPE2", obs_shape)
  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_shape,
      local_devices_to_use=local_devices_to_use,
      sac_network=sac_network,
      alpha_optimizer=alpha_optimizer,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
  )
  del global_key

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    training_state = training_state.replace(
        normalizer_params=_replicate_across_devices(
            params[0], local_devices_to_use
        ),
        policy_params=_replicate_across_devices(params[1], local_devices_to_use),
    )



  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use)
  )
  import copy
  eval_env = copy.deepcopy(environment)
  if evaluation_randomization_fn is not None and hasattr(environment, 'dr_range'):
    eval_dr_low, eval_dr_high = environment.dr_range
    eval_env = wrap_for_adv_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=functools.partial(
            evaluation_randomization_fn,
            dr_range=environment.dr_range,
        ),
        param_size=len(eval_dr_low),
        dr_range_low=eval_dr_low,
        dr_range_high=eval_dr_high,
    )
    evaluator = AdvEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        dr_range_low=eval_dr_low,
        dr_range_high=eval_dr_high,
    )
  else:
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
    )
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.policy_params)
        ),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # Create and initialize the replay buffer.
  t = time.time()
  prefill_key, local_key = jax.random.split(local_key)
  prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
  training_state, env_state, buffer_state, _ = prefill_replay_buffer(
      training_state, env_state, buffer_state, prefill_keys
  )

  replay_size = (
      jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
  )
  logging.info('replay size after prefill %s', replay_size)
  assert replay_size >= min_replay_size
  training_walltime = time.time() - t

  # 리턴 히트맵 함수 정의
  def log_performance_heatmap(ts, current_step):
    # 각 dr 도메인에서 '실제 에피소드 리턴'(seed 평균)을 격자로 시각화 (논문 Performance Heatmap)
    if process_id != 0 or eval_dr_low is None or len(eval_dr_low) != 2:
      return
    G = int(round(num_eval_envs ** 0.5))                     # 격자 = sqrt(num_eval_envs)
    xs = jnp.linspace(eval_dr_low[0], eval_dr_high[0], G)
    ys = jnp.linspace(eval_dr_low[1], eval_dr_high[1], G)
    gx, gy = jnp.meshgrid(xs, ys, indexing='ij')
    grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)     # (G*G,2) = num_eval_envs개 도메인
    _, reward_1d = evaluator.run_evaluation(
        _unpmap((ts.normalizer_params, ts.policy_params)),
        dynamics_params=grid, num_eval_seeds=5, return_reward_array=True)
    Z = np.asarray(reward_1d).reshape(G, G)
    fig, ax = plt.subplots()
    cs = ax.contourf(np.asarray(gx), np.asarray(gy), Z, levels=30, cmap='viridis')
    fig.colorbar(cs, label='episode return')
    ax.set_xlabel('dr param 0'); ax.set_ylabel('dr param 1')
    wandb.log({"performance_heatmap": wandb.Image(fig)}, step=int(current_step))
    plt.close(fig)

  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, env_state, buffer_state, training_metrics) = (
        training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
    )
    current_step = int(_unpmap(training_state.env_steps))

    # Eval and logging
    if process_id == 0:
      if checkpoint_logdir:
        params = _unpmap(
            (training_state.normalizer_params, training_state.policy_params)
        )
        ckpt_config = checkpoint.network_config(
            observation_size=obs_shape,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        checkpoint.save(checkpoint_logdir, current_step, params, ckpt_config)

      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)
          ),
          training_metrics,
      )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  log_performance_heatmap(training_state, current_step)
  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params)
  )

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
