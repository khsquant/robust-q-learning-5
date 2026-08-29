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
from typing import Any, Callable, Dict, Optional, Tuple, Union, NamedTuple, Sequence

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
from agents.td3 import checkpoint
from agents.td3 import losses as td3_losses
from agents.td3 import networks as td3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.envs.base import Wrapper
import flax
import jax
import jax.numpy as jnp
import optax
from brax.envs.base import Wrapper, Env, State
from brax.training.types import Policy, PolicyParams, PRNGKey, Metrics, Transition
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import Evaluator, AdvEvaluator
from learning.module.wrapper.wrapper import Wrapper
import wandb
import numpy as np
import matplotlib.pyplot as plt
from flax.core import FrozenDict

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'

class TransitionwithCritic(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  dynamics_params: jax.Array
  q_values : jax.Array
  # target_lnpdf: jax.Array
  extras: FrozenDict[str, Any]  # recommended

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
  normalizer_params: running_statistics.RunningStatisticsState
  noise_scales: jnp.ndarray

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


def _uint64_mod(step: types.UInt64, divisor: int) -> jax.Array:
  """Computes a small integer modulo for Brax UInt64 counters."""
  hi_mod = step.hi % divisor
  lo_mod = step.lo % divisor
  word_mod = (2**32) % divisor
  return (hi_mod * word_mod + lo_mod) % divisor


def _init_training_state(
    key: PRNGKey,
    obs_size: Union[int, Dict[str, specs.Array]],    
    local_devices_to_use: int,
    td3_network: td3_networks.Td3Networks,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    num_envs : int,
    std_max: float =0.4,
    std_min : float =0.05,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_q, key_noise = jax.random.split(key,3 )

  policy_params = td3_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = td3_network.q_network.init(key_q)
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
      normalizer_params=normalizer_params,
      noise_scales=jax.random.uniform(
          key_noise,
          (num_envs // local_devices_to_use // jax.process_count(),),
          minval=std_min,
          maxval=std_max,
      ),
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
    network_factory: types.NetworkFactory[
        td3_networks.Td3Networks
    ] = td3_networks.make_td3_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    eval_randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    dr_train_ratio = 1.0,
    std_max=0.4,
    std_min=0.05,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_frequency: int = 2,
    distributional_q=False,
    dr_augmented_critic: bool = False,
    use_wandb=False,
):
  """td3 training."""
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
  if policy_frequency < 1:
    raise ValueError('policy_frequency must be >= 1')

  if max_replay_size is None:
    max_replay_size = num_timesteps

  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of run_one_td3_epoch calls per run_td3_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
  # num_evals_after_init = 1000
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )
  print("local devices to us", local_devices_to_use)
  print("process count", jax.process_count())
  assert num_envs % device_count == 0
  import copy
  env = copy.deepcopy(environment)

  rng = jax.random.PRNGKey(seed)
  rng, key = jax.random.split(rng)
  evaluation_randomization_fn = eval_randomization_fn or randomization_fn
  
  if randomization_fn is not None:
    if not hasattr(env, 'dr_range'):
      raise ValueError(
          'TD3 domain randomization requires an environment with dr_range.'
      )
    dr_range_low, dr_range_high = env.dr_range
    dr_mid = (dr_range_low + dr_range_high) / 2.
    dr_scale = (dr_range_high - dr_range_low) / 2.
    training_dr_range = (
        dr_mid - dr_train_ratio * dr_scale,
        dr_mid + dr_train_ratio * dr_scale,
    )
    training_randomization_fn = functools.partial(
        randomization_fn,
        dr_range=training_dr_range,
    )
  else:
    dr_range_low = jnp.zeros((0,), dtype=jnp.float32)
    dr_range_high = jnp.zeros((0,), dtype=jnp.float32)
    training_randomization_fn = None

  param_size = len(dr_range_low)
  env = wrap_for_adv_training(
    env,
    episode_length=episode_length,
    action_repeat=action_repeat,
    randomization_fn=training_randomization_fn,
    param_size=param_size,
    dr_range_low=dr_range_low,
    dr_range_high=dr_range_high,
  )


  obs_shape = env.observation_size
#   if isinstance(obs_size, Dict):
#     obs_size = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    # raise NotImplementedError('Dictionary observations not implemented in td3')
  print("td3 OBS SIZE", obs_shape)
  action_size = env.action_size

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  td3_network, q_support = network_factory(
      observation_size=obs_shape,
      action_size=action_size,
      param_size=param_size,
      preprocess_observations_fn=normalize_fn,
      distributional_q=distributional_q,
      dr_augmented_critic=dr_augmented_critic,
  )
  make_policy = td3_networks.make_inference_fn(td3_network)


  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)

  dummy_obs = { key: jnp.zeros(obs_shape[key]) for key in obs_shape } if isinstance(obs_shape, dict) else jnp.zeros((obs_shape,))
  print("dummy_obs", dummy_obs)
  dummy_action = jnp.zeros((action_size,))
  dummy_q_values = (
      jnp.zeros((q_support.shape[0],), dtype=jnp.float32)
      if distributional_q
      else jnp.zeros((2,), dtype=jnp.float32)
  )
  dummy_transition = TransitionwithCritic(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      dynamics_params=jnp.zeros((param_size,), dtype=jnp.float32),
      q_values=dummy_q_values,
      # target_lnpdf=dummy_q_values,
      extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
  )
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  critic_loss, actor_loss = td3_losses.make_losses(
      td3_network=td3_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      distributional_q=distributional_q,
      dr_augmented_critic=dr_augmented_critic,
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: TransitionwithCritic
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry
    key, key_critic, key_actor,key_noise = jax.random.split(key, 4)
    noise = jax.random.normal(key_noise, shape=transitions.action.shape) * policy_noise
    noise = jnp.clip(noise,-noise_clip, noise_clip)
    if distributional_q:
      (critic_loss, (current_q, next_v)), q_params, q_optimizer_state = critic_update(
          training_state.q_params,
          training_state.policy_params,
          training_state.normalizer_params,
          training_state.target_q_params,
          transitions,
          noise,
          q_support,
          key_critic,
          optimizer_state=training_state.q_optimizer_state,
      )
    else:
      (critic_loss, (current_q, next_v)), q_params, q_optimizer_state = critic_update(
          training_state.q_params,
          training_state.policy_params,
          training_state.normalizer_params,
          training_state.target_q_params,
          transitions,
          noise,
          key_critic,
          optimizer_state=training_state.q_optimizer_state,
      )

    def polyak_update(target_params, params):
      return jax.tree_util.tree_map(
          lambda x, y: x * (1 - tau) + y * tau,
          target_params,
          params,
      )

    new_target_q_params = polyak_update(training_state.target_q_params, q_params)

    def update_actor(_):
      actor_loss, policy_params, policy_optimizer_state = actor_update(
          training_state.policy_params,
          training_state.normalizer_params,
          q_params,
          transitions,
          key_actor,
          optimizer_state=training_state.policy_optimizer_state,
      )
      return (
          actor_loss,
          policy_params,
          policy_optimizer_state,
      )

    def skip_actor(_):
      return (
          jnp.zeros_like(critic_loss),
          training_state.policy_params,
          training_state.policy_optimizer_state,
      )

    new_gradient_steps = training_state.gradient_steps + 1
    should_update_actor = _uint64_mod(new_gradient_steps, policy_frequency) == 0
    (
        actor_loss,
        policy_params,
        policy_optimizer_state,
    ) = jax.lax.cond(
        should_update_actor,
        update_actor,
        skip_actor,
        operand=None,
    )

    metrics = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'actor_updated': should_update_actor.astype(jnp.float32),
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
        gradient_steps=new_gradient_steps,
        env_steps=training_state.env_steps,
        normalizer_params=training_state.normalizer_params,
        noise_scales=training_state.noise_scales,
    )
    return (new_training_state, key), metrics
  def current_env_params(env_state: envs.State, reset_params) -> jax.Array:
    done = env_state.done[..., None]
    #return reset_params
    return env_state.info["dr_params"] * (1 - done) + reset_params * done

  def adv_step(
    env: Env,
    env_state: State,
    policy: Policy,
    normalizer_params,
    q_params,
    target_q_params,
    dynamics_params: jnp.ndarray,
    noise_scales : jnp.ndarray,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
  ):
    step_key, key = jax.random.split(key)
    actions, policy_extras = policy(env_state.obs, noise_scales, key)
    
    eff_params = current_env_params(env_state, dynamics_params)     # 에피소드 내 고정
    nstate = env.step(env_state, actions, eff_params)
    #nstate = env.step(env_state, actions, current_env_params(env_state, dynamics_params))
    
    if dr_augmented_critic:
      q_values = td3_network.q_network.apply(
          normalizer_params, q_params, env_state.obs, actions, dynamics_params
      ).mean(-1)
    else:
      q_values = td3_network.q_network.apply(
          normalizer_params, q_params, env_state.obs, actions
      ).mean(-1)
    # target_lnpdfs = jax.nn.log_softmax(-q_values, -1)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, TransitionwithCritic(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation= nstate.obs,
        #dynamics_params=dynamics_params,
        dynamics_params=eff_params,
        q_values = q_values,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
    )
  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      q_params: Params,
      target_q_params: Params,
      dynamics_params: jnp.ndarray,
      noise_scales: jnp.ndarray,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      envs.State,
      ReplayBufferState,
  ]:
    noise_key, key = jax.random.split(key)
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = adv_step(
        env, env_state, policy, normalizer_params, q_params, target_q_params, dynamics_params, noise_scales, key, extra_fields=('truncation',)
    )
    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
    noise_scales = (
        (1 - env_state.done) * noise_scales
        + env_state.done
        * jax.random.uniform(
            noise_key,
            shape=noise_scales.shape,
            minval=std_min,
            maxval=std_max,
        )
    )
    q_values = transitions.q_values
    # dynamics_params = jax.random.uniform(key=jax.random.PRNGKey(seed), shape=(noise_scales.shape[0],len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)
    simul_info ={
      "simul/reward_mean" : transitions.reward.mean(),
      "simul/reward_std" : transitions.reward.std(),
      "simul/reward_max" : transitions.reward.max(),
      "simul/reward_min" : transitions.reward.min(),
      "simul/q_values" : q_values.mean(),
      "simul/q_values_std" : q_values.std(),
      "simul/q_values_max" : q_values.max(),
      "simul/q_values_p75" : jnp.quantile(q_values, 0.75),
      "simul/q_values_p25" : jnp.quantile(q_values, 0.25),
      "simul/q_values_mid" : jnp.quantile(q_values, 0.5),
      "simul/q_values_min" : q_values.min(),

      # "simul/dynamics_params_mean" : dynamics_params.mean(),
      # "simul/dynamics_params_std" : dynamics_params.std(),
    }
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, noise_scales, env_state, buffer_state, simul_info, transitions

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
    experience_key, training_key, param_key = jax.random.split(key, 3)
    dynamics_params = jax.random.uniform(
        key=param_key,
        shape=env_state.info["dr_params"].shape,
        minval=dr_range_low,
        maxval=dr_range_high,
    )
    
    normalizer_params, noise_scales, env_state, buffer_state, simul_info, simul_transitions = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.q_params,
        training_state.target_q_params,
        dynamics_params,
        training_state.noise_scales,
        env_state,
        buffer_state,
        experience_key,
    )
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        noise_scales = noise_scales,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
    )

    buffer_state, transitions = replay_buffer.sample(buffer_state)
    # Change the front dimension of transitions so 'update_step' is called
    # grad_updates_per_step times by the scan.
    per_update_batch_size = batch_size // device_count
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, (grad_updates_per_step, per_update_batch_size) + x.shape[1:]
        ),
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
      key, new_key, step_key = jax.random.split(key,3)
      dynamics_params = jax.random.uniform(
          key=step_key,
          shape=env_state.info["dr_params"].shape,
          minval=dr_range_low,
          maxval=dr_range_high,
      )
      new_normalizer_params, new_noise_scales, env_state, buffer_state, simul_info, simul_transitions = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.q_params,
        training_state.target_q_params,
        dynamics_params,
        training_state.noise_scales,
        env_state,
        buffer_state,
        key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          noise_scales = new_noise_scales,
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
  def evaluation_on_current_occupancy(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
    # shape = np.sqrt(num_envs).astype(int)
    x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 32),\
                          jnp.linspace(dr_range_low[1], dr_range_high[1], 32))
    dynamics_params_grid = jnp.c_[x.ravel(), y.ravel()]
    def f(carry, dynamics_param):
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)

      new_normalizer_params, new_noise_scales, env_state, buffer_state, simul_info, simul_transitions = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          training_state.q_params,
          training_state.target_q_params,
          dynamics_param[None, ...] * jnp.ones_like(env_state.info["dr_params"]),
          training_state.noise_scales,
          env_state,
          buffer_state,
          key,
      )
      # pdf_values = simul_transitions.target_lnpdf
      pdf_values = simul_transitions.q_values # 수정됨
      
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          noise_scales = new_noise_scales,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), pdf_values
    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key), xs= dynamics_params_grid,
    )[1]

  evaluation_on_current_occupancy = jax.pmap(
      evaluation_on_current_occupancy, axis_name=_PMAP_AXIS_NAME
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
  local_key, rb_key, env_key, eval_key, reset_key = jax.random.split(local_key, 5)
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )
  env_state = jax.pmap(env.reset)(env_keys)
  print("obs", jax.tree_util.tree_map( lambda x: x.shape , env_state.obs))
  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs

  )
  print("td3 OBS SHAPE2", obs_shape)
  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_shape,
      local_devices_to_use=local_devices_to_use,
      td3_network=td3_network,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      num_envs=num_envs,
      std_max=std_max,
      std_min=std_min,
  )
  del global_key
  # Env init

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    training_state = training_state.replace(
        normalizer_params=_replicate_across_devices(
            params[0], local_devices_to_use
        ),
        policy_params=_replicate_across_devices(params[1], local_devices_to_use),
        noise_scales=_replicate_across_devices(params[2], local_devices_to_use),
    )


  # Replay buffer init
  buffer_state = jax.pmap(replay_buffer.init)(
      jax.random.split(rb_key, local_devices_to_use)
  )

  eval_env = copy.deepcopy(environment)
  eval_dr_low, eval_dr_high = dr_range_low, dr_range_high
  eval_randomization_wrapper_fn = None
  if evaluation_randomization_fn is not None:
    if not hasattr(environment, 'dr_range'):
      raise ValueError(
          'TD3 evaluation randomization requires an environment with dr_range.'
      )
    eval_dr_low, eval_dr_high = environment.dr_range
    eval_randomization_wrapper_fn = functools.partial(
        evaluation_randomization_fn,
        dr_range=environment.dr_range,
    )
  eval_env = wrap_for_adv_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=eval_randomization_wrapper_fn,
      param_size=param_size,
      dr_range_low=eval_dr_low,
      dr_range_high=eval_dr_high,
  )
  evaluator = AdvEvaluator(
      eval_env,
      functools.partial(make_policy, deterministic=True),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
      dr_range_low=eval_dr_low,
      dr_range_high=eval_dr_high,
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
  #evaluation on current occupancy
  if  len(dr_range_low)==2:
    evaluation_key, local_key = jax.random.split(local_key)
    evaluation_key = jax.random.split(evaluation_key, local_devices_to_use)
    target_pdfs = evaluation_on_current_occupancy(
        training_state, env_state, buffer_state, evaluation_key
    )
    print("target_pdfs", target_pdfs.shape)
    shape = np.sqrt(num_envs).astype(int)
    x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 32),\
                          jnp.linspace(dr_range_low[1], dr_range_high[1], 32))
    target_pdfs = target_pdfs.mean(axis=(0,2))
    target_pdfs = jnp.reshape(target_pdfs, x.shape)
    print("x shape", x.shape)
    if process_id==0:
      target_fig = plt.figure()
      ctf = plt.contourf(x, y, target_pdfs, levels=20, cmap='viridis')
      cbar = target_fig.colorbar(ctf)
      if use_wandb:
        wandb.log({
          'target_prob on current occupancy with critic' : wandb.Image(target_fig)
        }, step=0)
  training_walltime = time.time() - t

  current_step = 0
  for _ in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    epoch_key, evaluation_key, local_key = jax.random.split(local_key, 3)
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
            (training_state.normalizer_params, training_state.policy_params, training_state.noise_scales)
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
      #evaluation on current occupancy
    evaluation_key = jax.random.split(evaluation_key, local_devices_to_use)
    if len(dr_range_low)==2:
      target_pdfs = evaluation_on_current_occupancy(
          training_state, env_state, buffer_state, evaluation_key
      )
      x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 32),\
                            jnp.linspace(dr_range_low[1], dr_range_high[1], 32))
      target_pdfs = target_pdfs.mean(axis=(0,2))
      target_pdfs = jnp.reshape(target_pdfs, x.shape)
      target_fig = plt.figure()
      ctf = plt.contourf(x, y, target_pdfs, levels=20, cmap='viridis')
      cbar = target_fig.colorbar(ctf)
      if use_wandb:
        wandb.log({
          'target_prob on current occupancy with critic' : wandb.Image(target_fig)
        }, step=int(current_step))
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
