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
from agents.m2td3 import checkpoint
from agents.m2td3 import losses as m2td3_losses
from agents.m2td3 import networks as m2td3_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.envs.base import Wrapper
import flax
import jax
import jax.numpy as jnp
import optax
from brax.envs.base import Wrapper, Env, State
from brax.training.types import Policy, PolicyParams, PRNGKey, Metrics
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import Evaluator, AdvEvaluator
from learning.module.wrapper.wrapper import Wrapper
from flax.core import FrozenDict

Metrics = types.Metrics
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'
class TransitionwithParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: FrozenDict[str, Any]  # recommended

@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  omega_optimizer_state: optax.OptState
  omega_params: Params
  omega_prob: jnp.ndarray
  active_omega: jnp.ndarray
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
    per_replica_batch: int,
    m2td3_network: m2td3_networks.M2TD3Networks,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    omega_optimizer: optax.GradientTransformation,
    num_envs : int,
    param_dim : int,
    dr_range_low,
    dr_range_high,
    std_max: float =0.4,
    std_min : float =0.05,
    num_omegas: int = 5,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_q, key_noise, key_omega, key_active_omega = jax.random.split(key, 5)
  del per_replica_batch
  per_device_envs = num_envs // local_devices_to_use // jax.process_count()
  omega_params = jax.random.uniform(
      key_omega,
      shape=(num_omegas, param_dim),
      minval=dr_range_low,
      maxval=dr_range_high,
  )
  omega_prob = jnp.ones((num_omegas,), dtype=jnp.float32) / num_omegas
  active_omega = jax.random.uniform(
      key_active_omega,
      shape=(per_device_envs, param_dim),
      minval=dr_range_low,
      maxval=dr_range_high,
  )
  
  omega_optimizer_state = omega_optimizer.init(omega_params)

  policy_params = m2td3_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = m2td3_network.q_network.init(key_q)
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
      omega_optimizer_state=omega_optimizer_state,
      omega_params=omega_params,
      omega_prob=omega_prob,
      active_omega=active_omega,
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
        m2td3_networks.M2TD3Networks
    ] = m2td3_networks.make_m2td3_networks,
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
    noise_clip=0.5,
    policy_noise = 0.2,
    omega_noise_rate : float = 0.2,
    omega_std : float = 0.3, # 원래1.0 
    omega_clip : float = 0.5,
    num_omegas : int = 5,
    omega_distance_threshold: float = 0.1,
    omega_lr: Optional[float] = None,
    omega_min_probability: float = 5e-2,
    omega_prob_update_rate: Optional[float] = None,
    omega_restart_distance: bool = True,
    omega_restart_probability: bool = True,
    policy_frequency: int = 2,
    dr_augmented_critic: bool = True,
):
  """m2td3 training."""
  if not dr_augmented_critic:
    raise ValueError("M2TD3 requires dr_augmented_critic=true for Q(s, a, omega).")
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
  # The number of run_one_m2td3_epoch calls per run_m2td3_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
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
  
  if hasattr(env,'dr_range') :
    dr_low, dr_high = env.dr_range
    dr_mid = (dr_low + dr_high) / 2.
    dr_scale = (dr_high - dr_low) / 2.
    training_dr_range = (dr_mid - dr_train_ratio*dr_scale, dr_mid + dr_train_ratio*dr_scale)
    dr_range_low, dr_range_high = training_dr_range
  else:
    raise ValueError("Environment does not have dr_range attribute. Please provide a valid environment with dr_range.")
  training_randomization_fn = None
  omega_prob_update_rate = (
      1.0 / max(episode_length, 1)
      if omega_prob_update_rate is None
      else omega_prob_update_rate
  )
  env = wrap_for_adv_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=functools.partial(randomization_fn,dr_range=training_dr_range),
      param_size=len(dr_range_low),
      dr_range_low=dr_range_low,
      dr_range_high=dr_range_high,
  )  # pytype: disable=wrong-keyword-args

  obs_shape = env.observation_size
#   if isinstance(obs_size, Dict):
#     obs_size = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    # raise NotImplementedError('Dictionary observations not implemented in m2td3')
  print("m2td3 OBS SIZE", obs_shape)
  action_size = env.action_size

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  m2td3_network = network_factory(
      observation_size=obs_shape,
      action_size=action_size,
      param_size=len(dr_range_low),
      preprocess_observations_fn=normalize_fn,
  )
  make_policy = m2td3_networks.make_inference_fn(m2td3_network)


  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)
  omega_optimizer = optax.adam(
      learning_rate=learning_rate if omega_lr is None else omega_lr
  )

  dummy_obs = { key: jnp.zeros(obs_shape[key]) for key in obs_shape } if isinstance(obs_shape, dict) else jnp.zeros((obs_shape,))
  print("dummy_obs", dummy_obs)
  dummy_action = jnp.zeros((action_size,))
  dummy_params = jnp.zeros((len(dr_range_low),))  # Dummy dynamics parameters
  dummy_transition = TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      dynamics_params=dummy_params,
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

  critic_loss, actor_loss, omega_loss = m2td3_losses.make_losses(
      m2td3_network=m2td3_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )
  omega_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      omega_loss, omega_optimizer,  has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: TransitionwithParams
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry

    key, key_critic, key_actor, key_omega, key_noise = jax.random.split(key, 5)
    noise = jax.random.normal(key_noise, shape=transitions.action.shape) * policy_noise
    noise = jnp.clip(noise,-noise_clip, noise_clip)
    (critic_loss, (current_q, next_v)), q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        transitions,
        noise,
        dr_range_low,
        dr_range_high,
        omega_noise_rate,
        omega_clip,
        key_critic,
        optimizer_state=training_state.q_optimizer_state,
    )

    def polyak_update(target_params, params):
      return jax.tree_util.tree_map(
          lambda x, y: x * (1 - tau) + y * tau,
          target_params,
          params,
      )

    new_gradient_steps = training_state.gradient_steps + 1
    should_update_actor = _uint64_mod(new_gradient_steps, policy_frequency) == 0
    new_target_q_params = polyak_update(training_state.target_q_params, q_params)

    def update_actor_and_omega(_):
      (
          omega_loss,
          (worst_omega_idx, worst_policy_loss),
      ), omega_params, omega_optimizer_state = omega_update(
          training_state.omega_params,
          training_state.policy_params,
          training_state.normalizer_params,
          q_params,
          transitions,
          key_omega,
          optimizer_state=training_state.omega_optimizer_state,
      )
      omega_params = jnp.clip(omega_params, dr_range_low, dr_range_high)
      actor_dynamics_params = training_state.omega_params[worst_omega_idx]
      actor_loss, policy_params, policy_optimizer_state = actor_update(
          training_state.policy_params,
          training_state.normalizer_params,
          q_params,
          actor_dynamics_params,
          transitions,
          key_actor,
          optimizer_state=training_state.policy_optimizer_state,
      )
      omega_prob = update_omega_prob(training_state.omega_prob, worst_omega_idx)
      return (
          actor_loss,
          omega_loss,
          worst_policy_loss,
          policy_params,
          policy_optimizer_state,
          omega_params,
          omega_optimizer_state,
          omega_prob,
      )

    def skip_actor_and_omega(_):
      return (
          jnp.zeros_like(critic_loss),
          jnp.zeros_like(critic_loss),
          jnp.zeros_like(critic_loss),
          training_state.policy_params,
          training_state.policy_optimizer_state,
          training_state.omega_params,
          training_state.omega_optimizer_state,
          training_state.omega_prob,
      )

    (
        actor_loss,
        omega_loss,
        worst_policy_loss,
        policy_params,
        policy_optimizer_state,
        omega_params,
        omega_optimizer_state,
        omega_prob,
    ) = jax.lax.cond(
        should_update_actor,
        update_actor_and_omega,
        skip_actor_and_omega,
        operand=None,
    )

    metrics = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'actor_updated': should_update_actor.astype(jnp.float32),
        'omega_loss': omega_loss,
        'omega_worst_policy_loss': worst_policy_loss,
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
        omega_optimizer_state=omega_optimizer_state,
        omega_params=omega_params,
        omega_prob=omega_prob,
        active_omega=training_state.active_omega,
        gradient_steps=new_gradient_steps,
        env_steps=training_state.env_steps,
        normalizer_params=training_state.normalizer_params,
        noise_scales=training_state.noise_scales,
    )
    return (new_training_state, key), metrics


  def adv_step(
    env: Env,
    env_state: State,
    dynamics_params: jnp.ndarray,
    policy: Policy,
    noise_scales : jnp.ndarray,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
  ):
    step_key, key = jax.random.split(key)
    actions, policy_extras = policy(env_state.obs, noise_scales, key)
    # dynamics_params = jax.random.uniform(key=step_key, shape=(num_envs,len(dr_range_low)), minval=dr_range_low, maxval=dr_range_high)
    nstate = env.step(env_state, actions, dynamics_params)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        dynamics_params=dynamics_params,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation= nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
    )
  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      noise_scales: jnp.ndarray,
      env_state: envs.State,
      dynamics_params: jnp.ndarray, #[num_envs(local), dynamics_param_size]
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
        env, env_state, dynamics_params, policy, noise_scales, key, extra_fields=('truncation',)
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

    simul_info ={
      "simul/reward_mean" : transitions.reward.mean(),
      "simul/reward_std" : transitions.reward.std(),
      "simul/reward_max" : transitions.reward.max(),
      "simul/reward_min" : transitions.reward.min(),
      "simul/dynamics_params_mean" : dynamics_params.mean(),
      "simul/dynamics_params_std" : dynamics_params.std(),

    }
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    return normalizer_params, noise_scales, env_state, buffer_state, simul_info
  def normalize_omega_prob(omega_prob):
    omega_prob = jnp.maximum(omega_prob, 0.0)
    total = jnp.sum(omega_prob)
    safe_total = jnp.maximum(total, 1e-8)
    uniform = jnp.ones_like(omega_prob) / omega_prob.shape[0]
    return jnp.where(total > 0, omega_prob / safe_total, uniform)

  def restart_omegas(omega_params, omega_prob, key):
    if num_omegas <= 1:
      return (
          omega_params,
          jnp.ones_like(omega_prob),
          jnp.array(0.0, dtype=jnp.float32),
          jnp.array(0.0, dtype=jnp.float32),
      )

    distance_mask = jnp.zeros((num_omegas,), dtype=bool)
    if omega_restart_distance:
      l1 = jnp.sum(
          jnp.abs(omega_params[:, None, :] - omega_params[None, :, :]),
          axis=-1,
      )
      close_upper = jnp.triu(
          (l1 <= omega_distance_threshold),
          k=1,
      )
      distance_mask = jnp.any(close_upper, axis=1)

    omega_prob = normalize_omega_prob(omega_prob)
    probability_mask = jnp.zeros((num_omegas,), dtype=bool)
    if omega_restart_probability:
      probability_mask = omega_prob < omega_min_probability

    restart_mask = distance_mask | probability_mask
    new_omega_params = jax.random.uniform(
        key,
        omega_params.shape,
        minval=dr_range_low,
        maxval=dr_range_high,
    )
    omega_params = jnp.where(restart_mask[:, None], new_omega_params, omega_params)

    restart_count = jnp.sum(restart_mask)
    kept_count = num_omegas - restart_count
    kept_mass = jnp.sum(jnp.where(restart_mask, 0.0, omega_prob))
    replacement_prob = jnp.where(
        kept_count > 0,
        kept_mass / jnp.maximum(kept_count, 1),
        1.0 / num_omegas,
    )
    omega_prob = jnp.where(restart_mask, replacement_prob, omega_prob)
    omega_prob = normalize_omega_prob(omega_prob)
    return (
        omega_params,
        omega_prob,
        jnp.sum(distance_mask).astype(jnp.float32),
        jnp.sum(probability_mask).astype(jnp.float32),
    )

  def update_omega_prob(omega_prob, selected_idx):
    if num_omegas <= 1:
      return jnp.ones_like(omega_prob)
    one_hot = jax.nn.one_hot(selected_idx, num_omegas, dtype=omega_prob.dtype)
    omega_prob = (
        omega_prob * (1.0 - omega_prob_update_rate)
        + omega_prob_update_rate * one_hot
    )
    return normalize_omega_prob(omega_prob)

  def sample_omega_batch(omega_params, omega_prob, key, noise_key, batch_count):
    omega_prob = normalize_omega_prob(omega_prob)
    selected_idx = jax.random.choice(
        key,
        num_omegas,
        shape=(batch_count,),
        p=omega_prob,
    )
    dynamics_params = omega_params[selected_idx]
    dynamics_params = dynamics_params + (
        omega_std
        * (dr_range_high - dr_range_low)
        / 2.0
        * jax.random.normal(noise_key, dynamics_params.shape)
    )
    dynamics_params = jnp.clip(dynamics_params, dr_range_low, dr_range_high)
    return dynamics_params, selected_idx

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
    experience_key, training_key, restart_key, sample_key, sample_noise_key = (
        jax.random.split(key, 5)
    )
    dynamics_params = training_state.active_omega
    normalizer_params, noise_scales, env_state, buffer_state, simul_info = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.noise_scales,
        env_state,
        dynamics_params,
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
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )
    done_mask = env_state.done.astype(bool)
    has_done = jnp.any(done_mask)

    def restart_for_done(_):
      return restart_omegas(
          training_state.omega_params,
          training_state.omega_prob,
          restart_key,
      )

    def skip_restart(_):
      return (
          training_state.omega_params,
          normalize_omega_prob(training_state.omega_prob),
          jnp.array(0.0, dtype=jnp.float32),
          jnp.array(0.0, dtype=jnp.float32),
      )

    omega_params, omega_prob, distance_restarts, probability_restarts = jax.lax.cond(
        has_done,
        restart_for_done,
        skip_restart,
        operand=None,
    )
    sampled_omega, sampled_idx = sample_omega_batch(
        omega_params,
        omega_prob,
        sample_key,
        sample_noise_key,
        training_state.active_omega.shape[0],
    )
    active_omega = jnp.where(done_mask[:, None], sampled_omega, training_state.active_omega)
    training_state = training_state.replace(
        omega_params=omega_params,
        omega_prob=omega_prob,
        active_omega=active_omega,
    )

    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
    metrics['omega_prob_min'] = omega_prob.min()
    metrics['omega_prob_max'] = omega_prob.max()
    metrics['omega_distance_restarts'] = distance_restarts
    metrics['omega_probability_restarts'] = probability_restarts
    metrics['omega_resampled_envs'] = done_mask.sum().astype(jnp.float32)
    metrics['omega_sampled_idx_mean'] = sampled_idx.astype(jnp.float32).mean()
    metrics.update(simul_info)
    return training_state, env_state, buffer_state, metrics

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, params):
      training_state, env_state, buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, new_noise_scales, env_state, buffer_state, simul_info = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          training_state.noise_scales,
          env_state,
          params,
          buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          noise_scales = new_noise_scales,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), ()
    param_key, key = jax.random.split(key)
    dynamics_params = jax.random.uniform(
      param_key, shape=(num_prefill_actor_steps * num_envs // jax.process_count() // local_devices_to_use, len(dr_range_low),),
        minval=dr_range_low, maxval=dr_range_high
      )

    dynamics_params = jnp.reshape(
        dynamics_params, (num_prefill_actor_steps, num_envs // jax.process_count() // local_devices_to_use) + dynamics_params.shape[1:]
    )
    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        dynamics_params,
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
  local_key, rb_key, env_key, eval_key, param_key = jax.random.split(local_key, 5)
  env_keys = jax.random.split(env_key, num_envs // jax.process_count())
  env_keys = jnp.reshape(
      env_keys, (local_devices_to_use, -1) + env_keys.shape[1:]
  )

  env_state = jax.pmap(env.reset)(env_keys)
  print("obs", jax.tree_util.tree_map( lambda x: x.shape , env_state.obs))
  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs

  )
  print("m2td3 OBS SHAPE2", obs_shape)
  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_shape,
      local_devices_to_use=local_devices_to_use,
      per_replica_batch= batch_size  // device_count,
      m2td3_network=m2td3_network,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      omega_optimizer=omega_optimizer,
      num_envs=num_envs,
      param_dim=len(dr_range_low),
      dr_range_low=dr_range_low,
      dr_range_high=dr_range_high,
      std_max=std_max,
      std_min=std_min,
      num_omegas=num_omegas,
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
  evaluation_randomization_fn = eval_randomization_fn or randomization_fn
  if evaluation_randomization_fn is not None:
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
        functools.partial(make_policy, deterministic=True),
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
        functools.partial(make_policy, deterministic=True),
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
