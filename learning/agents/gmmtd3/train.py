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

import wandb
from learning.module.gmmvi.network import GMMTrainingState
from learning.module.wrapper.adv_wrapper import wrap_for_adv_training
from learning.module.wrapper.evaluator import Evaluator, AdvEvaluator
import learning.module.gmmvi.utils as gmm_utils

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
from agents.gmmtd3 import checkpoint
from agents.gmmtd3 import losses as gmmtd3_losses
from agents.gmmtd3 import networks as gmmtd3_networks
from agents.tcrmdp import common as tc_common
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.envs.base import Wrapper
import flax
import jax
import jax.numpy as jnp
import optax
from brax.envs.base import Wrapper, Env, State
from brax.training.types import Policy, PolicyParams, PRNGKey, Metrics
from flax.core import FrozenDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from learning.module.wrapper.wrapper import Wrapper
from matplotlib.ticker import MaxNLocator
Metrics = types.Metrics
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'

def target_contour_plot(samples, log_prob):
  xy = np.asarray(samples)
  x, y = xy[:, 0], xy[:,1]
  z = np.asarray(log_prob)
  m = np.isfinite(z)
  x, y, z = x[m], y[m], z[m]

  tri = mtri.Triangulation(x, y)
  fig, ax = plt.subplots()
  cs = plt.tricontourf(tri, z, levels=30, cmap='viridis')  # or tricontour for lines
  plt.colorbar(cs, label='log p(x)')
  plt.scatter(x, y, s=5, c='k', alpha=0.3)
  ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
  ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
  ax.minorticks_on()
  ax.grid(True, which='major', alpha=0.25, linewidth=0.8)
  ax.grid(True, which='minor', alpha=0.12, linewidth=0.6)

  return fig

class TransitionwithGMMParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  dynamics_params: jax.Array
  q_values : jax.Array
  target_lnpdf: jax.Array
  target_lnpdf_grad: jax.Array
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
  gmm_training_state : GMMTrainingState
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
    gmmtd3_network: gmmtd3_networks.GMMTd3Networks,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    gmm_init_state: GMMTrainingState,
    num_envs : int,
    std_max: float =0.4,
    std_min : float =0.05,
) -> TrainingState:
  """Inits the training state and replicates it over devices."""
  key_policy, key_q, key_noise = jax.random.split(key, 3)

  policy_params = gmmtd3_network.policy_network.init(key_policy)
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = gmmtd3_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  normalizer_params = running_statistics.init_state(
    #   specs.Array((obs_size,), jnp.dtype('float32'))
    #obs_size if isinstance(obs_size, dict) else specs.Array((obs_size,), jnp.dtype('float32'))
    obs_size # 수정됨
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
      gmm_training_state=gmm_init_state,
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
        gmmtd3_networks.GMMTd3Networks
    ] = gmmtd3_networks.make_gmmtd3_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    v_randomization_fn: Optional[
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
    eval_with_training_env=False,
    use_wandb=False,
    noise_clip=0.5,
    policy_noise = 0.2,
    policy_frequency: int = 2,
    value_obs_key="state",
    distributional_q=False,
    dr_augmented_critic: bool = False,
    beta: float = 1.0,
    use_tc: bool = False,
    radius: float = 0.001,
):
  """gmmtd3 training."""
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
  # num_evals=(num_timesteps - num_prefill_env_steps)//env_steps_per_actor_step
  if max_replay_size is None:
    max_replay_size = num_timesteps
  st = time.time()
  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_replay_size // num_envs)
  num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
  assert num_timesteps - num_prefill_env_steps >= 0
  num_evals_after_init = max(num_evals - 1, 1)
  # num_evals_after_init = 100
  # The number of run_one_td3_epoch calls per run_td3_training.
  # equals to
  # ceil(num_timesteps - num_prefill_env_steps /
  #      (num_evals_after_init * env_steps_per_actor_step))
  num_training_steps_per_epoch = -(
      -(num_timesteps - num_prefill_env_steps)
      // (num_evals_after_init * env_steps_per_actor_step)
  )
  WARM_FRAC = 0.3     # 각 환경 학습의 앞 30%에 걸쳐 β를 0→목표로 램프 (스케일-프리)
  warm_updates = WARM_FRAC * (num_timesteps - num_prefill_env_steps) / env_steps_per_actor_step
  
  print("num training steps per epoch", num_training_steps_per_epoch)
  assert num_envs % device_count == 0
  import copy
  env = copy.deepcopy(environment)

  rng = jax.random.PRNGKey(seed)
  rng, init_key = jax.random.split(rng)


  obs_shape = env.observation_size
  print("gmmtd3 OBS SIZE", obs_shape)
  action_size = env.action_size
  evaluation_randomization_fn = eval_randomization_fn or randomization_fn
  if hasattr(env, 'dr_range'):
    dr_range_low, dr_range_high = env.dr_range
    dr_mid = (dr_range_low + dr_range_high) / 2.0
    dr_scale = (dr_range_high - dr_range_low) / 2.0
    training_dr_range = (
        dr_mid - dr_train_ratio * dr_scale,
        dr_mid + dr_train_ratio * dr_scale,
    )
    dr_range_low, dr_range_high = training_dr_range
  else:
    raise ValueError(
        "GMMTD3 domain randomization requires an environment with dr_range."
    )
  dynamics_param_size = len(dr_range_low)
  can_visualize_dr = dynamics_param_size == 2
  print("dr_range_low", dr_range_low)
  print("dr_range_high", dr_range_high)
  training_randomization_fn = functools.partial(
      randomization_fn,
      dr_range=training_dr_range,
  )
  env = wrap_for_adv_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=training_randomization_fn,
      param_size = dynamics_param_size,
      dr_range_low = dr_range_low,
      dr_range_high= dr_range_high,
  )  # pytype: disable=wrong-keyword-args
  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  gmm_batch_size=4096
  gmmtd3_network, gmm_init_state = network_factory(
      observation_size=obs_shape,
      action_size=action_size,
      dynamics_param_size = dynamics_param_size,
      param_size=dynamics_param_size,
      batch_size= gmm_batch_size,
      num_envs = num_envs//jax.process_count(),
      init_key = init_key,
      preprocess_observations_fn=normalize_fn,
      bound_info = (dr_range_low, dr_range_high),
      dr_augmented_critic=dr_augmented_critic,
  )
  make_policy = gmmtd3_networks.make_inference_fn(gmmtd3_network)
  N_REF = min(64, num_envs)
  ref_rng = jax.random.split(jax.random.PRNGKey(0), num_envs)
  ref_obs = env.reset(ref_rng).obs[:N_REF]

  policy_optimizer = optax.adam(learning_rate=learning_rate)
  q_optimizer = optax.adam(learning_rate=learning_rate)    

  dummy_params = jnp.zeros((dynamics_param_size,))  # Dummy dynamics parameters
  dummy_obs = { key: jnp.zeros(obs_shape[key]) for key in obs_shape } if isinstance(obs_shape, dict) else jnp.zeros((obs_shape,))
  # dummy_model_grad = { key: jnp.zeros(obs_shape[key] + (dynamics_param_size,)) for key in obs_shape } if isinstance(obs_shape, dict) else jnp.zeros((obs_shape,))
  print("dummy_obs", dummy_obs)
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = TransitionwithGMMParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      dynamics_params=dummy_params,
      q_values=0.,
      target_lnpdf=0.,
      target_lnpdf_grad=dummy_params,

      extras={'state_extras': {'truncation': 0.0}, 'policy_extras': {}},
  )
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=batch_size * grad_updates_per_step // device_count,
  )

  critic_loss, actor_loss, gmm_update = gmmtd3_losses.make_losses(
      gmmtd3_network=gmmtd3_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
      dr_augmented_critic=dr_augmented_critic,
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer, has_aux=True, pmap_axis_name=_PMAP_AXIS_NAME
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
  )

  def sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: TransitionwithGMMParams
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    training_state, key = carry

    key, key_critic, key_actor,key_noise = jax.random.split(key, 4)
    
    noise = jax.random.normal(key_noise, shape=transitions.action.shape) * policy_noise
    noise = jnp.clip(noise,-noise_clip, noise_clip)
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

    new_training_state = training_state.replace(
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

  def current_env_params(env_state: envs.State, key: PRNGKey) -> jax.Array:
    reset_params = jax.random.uniform(
        key,
        shape=env_state.info["dr_params"].shape,
        minval=dr_range_low,
        maxval=dr_range_high,
    )
    done = env_state.done[..., None]
    return env_state.info["dr_params"] * (1 - done) + reset_params * done

  def adv_step(
    env: Env,
    env_state: State,
    policy: Policy,
    normalizer_params,
    q_params,
    target_q_params,
    dynamics_params,
    noise_scales : jnp.ndarray,
    key: PRNGKey,
    beta_scale=1.0,
    extra_fields: Sequence[str] = (),
  ):
    action_key, tc_key = jax.random.split(key)
    actions, policy_extras = policy(env_state.obs, noise_scales, action_key)
    effective_dynamics_params = dynamics_params
    if use_tc:
      effective_dynamics_params = tc_common.clip_params_to_radius(
          current_env_params(env_state, tc_key),
          dynamics_params,
          dr_range_low,
          dr_range_high,
          radius,
      )
    
    done = env_state.done[..., None]
    xi_env = env_state.info["dr_params"] * (1 - done) + dynamics_params * done  # 에피소드 내 고정
    nstate = env.step(env_state, actions, xi_env)
    
    #nstate = env.step(env_state, actions, effective_dynamics_params)
    state_extras = {x: nstate.info[x] for x in extra_fields} 

    if dr_augmented_critic:
      q_values = gmmtd3_network.q_network.apply(
          normalizer_params, q_params, env_state.obs, actions, dynamics_params #effective_dynamics_params
      ).mean(-1)
    else:
      q_values = gmmtd3_network.q_network.apply(
          normalizer_params, q_params, env_state.obs, actions
      ).mean(-1)
    # target_lnpdf = beta * q_values/100
    q_sg = jax.lax.stop_gradient(q_values)
    logw = (beta * beta_scale) * (q_sg - q_sg.mean()) / 100.0          # 배치 평균 차감 → 절대 Q 드리프트(음수 발산) 면역
    target_lnpdf = jnp.clip(logw, -3.0, 3.0)            # 집중 상한: 최대 가중비 e^6≈400배
    return nstate, TransitionwithGMMParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation= nstate.obs,
        #dynamics_params=effective_dynamics_params,
        dynamics_params=xi_env,
        q_values = q_values,
        target_lnpdf= target_lnpdf,
        target_lnpdf_grad= jnp.zeros_like(dynamics_params),
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
    )
  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      q_params : Params,
      target_q_params : Params,
      dynamics_params: Params,
      noise_scales: jnp.ndarray,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
      beta_scale=1.0,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      envs.State,
      ReplayBufferState,
  ]:
    noise_key, key = jax.random.split(key)
    policy = make_policy((normalizer_params, policy_params))
    env_state, transitions = adv_step(
        env,
        env_state,
        policy,
        normalizer_params,
        q_params,
        target_q_params,
        dynamics_params,
        noise_scales,
        key,
        beta_scale=beta_scale,
        extra_fields=('truncation',),
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
    # q_values = gmmtd3_network.q_network.apply(normalizer_params, target_q_params, transitions.observation, transitions.action).mean(-1)
    simul_info ={
      "simul/reward_mean" : transitions.reward.mean(),
      "simul/reward_std" : transitions.reward.std(),
      "simul/reward_max" : transitions.reward.max(),
      "simul/reward_min" : transitions.reward.min(),
      "simul/dynamics_params_mean" : dynamics_params.mean(),
      "simul/dynamics_params_std" :dynamics_params.std(),
      "simul/q_values" : q_values.mean(),
      "simul/q_values_std" : q_values.std(),
      "simul/q_values_max" : q_values.max(),
      "simul/q_values_p75" : jnp.nanquantile(q_values, 0.75),
      "simul/q_values_p25" : jnp.nanquantile(q_values, 0.25),
      "simul/q_values_mid" : jnp.nanquantile(q_values, 0.5),
      "simul/q_values_min" : q_values.min(),
    }
    # print("transitions", transitions)
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
    experience_key, training_key, param_key, key_gmm = jax.random.split(key, 4)
    sampled_dynamics_params, mapping = gmmtd3_network.gmm_network.sample_selector.select_samples(
        training_state.gmm_training_state.model_state,
        param_key,
    )

    nu = training_state.gmm_training_state.num_updates
    beta_scale = jnp.clip(nu / warm_updates, 0.0, 1.0)     # 0→1 램프 후 1 유지
    normalizer_params, noise_scales, env_state, buffer_state, simul_info, simul_transitions = get_experience(
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.q_params,
        training_state.target_q_params,
        sampled_dynamics_params,
        training_state.noise_scales,
        env_state,
        buffer_state,
        experience_key,
    )

    # (A') 메인 q_network(ξ 조건부)로 per-ξ 타깃 추정 — dr_augmented_critic=True 필수
    det_policy = make_policy((training_state.normalizer_params, training_state.policy_params),
                             deterministic=True)
    a0 = det_policy(ref_obs, jnp.zeros((ref_obs.shape[0], action_size)), param_key)[0]  # noise=0
    
    def _jhat(xi):
      xi_b = jnp.broadcast_to(xi, (ref_obs.shape[0],) + xi.shape)
      v0 = gmmtd3_network.q_network.apply(
          training_state.normalizer_params, training_state.q_params,
          ref_obs, a0, xi_b).mean(-1)
      return v0.mean()
    
    Jhat = jax.vmap(_jhat)(sampled_dynamics_params)                  # (num_xi,)
    logw = beta_scale * beta * (Jhat - Jhat.mean()) / 100.0
    gmm_target_lnpdf = jnp.clip(logw, -3.0, 3.0)
    gmm_target_grad = jnp.zeros_like(sampled_dynamics_params)
    
    new_sample_db_state = gmmtd3_network.gmm_network.sample_selector.save_samples(training_state.gmm_training_state.model_state, \
                      training_state.gmm_training_state.sample_db_state, sampled_dynamics_params, gmm_target_lnpdf, \
                        gmm_target_grad, mapping) # simul_transitions.dynamics_params -> sampled_dynamics_params
    new_gmm_training_state = training_state.gmm_training_state._replace(sample_db_state=new_sample_db_state)
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        noise_scales = noise_scales,
        gmm_training_state = new_gmm_training_state,
        env_steps=training_state.env_steps + env_steps_per_actor_step,
    )
    buffer_state, transitions = replay_buffer.sample(buffer_state)
    target_lnpdf = transitions.target_lnpdf
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        sgd_step, (training_state, training_key), transitions
    )
    new_gmm_training_state = gmm_update(training_state.gmm_training_state, key_gmm)
    training_state = training_state.replace(gmm_training_state=new_gmm_training_state)
    gmm_metrics={
        'num_components' : new_gmm_training_state.model_state.gmm_state.num_components,
        'gmm_mean_x_min' : new_gmm_training_state.model_state.gmm_state.means[:,0].min(),
        'gmm_mean_x_mean' : new_gmm_training_state.model_state.gmm_state.means[:,0].mean(),
        'gmm_mean_x_max' : new_gmm_training_state.model_state.gmm_state.means[:,0].max(),
        'gmm_mean_y_min' : new_gmm_training_state.model_state.gmm_state.means[:,1].min(),
        'gmm_mean_y_mean' : new_gmm_training_state.model_state.gmm_state.means[:,1].mean(),
        'gmm_mean_y_max' : new_gmm_training_state.model_state.gmm_state.means[:,1].max(),
    }
    metrics.update(gmm_metrics)
    metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
    metrics.update(simul_info)
    return training_state, env_state, buffer_state, metrics, sampled_dynamics_params, target_lnpdf # simul_transitions.dynamics_params -> sampled_dynamics_params

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, key = carry
      key, new_key, step_key = jax.random.split(key, 3)
      sampled_dynamics_params, mapping = gmmtd3_network.gmm_network.sample_selector.select_samples(
          training_state.gmm_training_state.model_state,
          step_key,
      )
      new_normalizer_params, new_noise_scales, env_state, buffer_state, simul_info, simul_transitions = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          training_state.q_params,
          training_state.target_q_params,
          sampled_dynamics_params,
          training_state.noise_scales,
          env_state,
          buffer_state,
          key,
          beta_scale=0.0,
      )
      new_sample_db_state = gmmtd3_network.gmm_network.sample_selector.save_samples(training_state.gmm_training_state.model_state, \
                      training_state.gmm_training_state.sample_db_state, sampled_dynamics_params, simul_transitions.target_lnpdf, \
                        simul_transitions.target_lnpdf_grad, mapping) # simul_transitions.dynamics_params -> sampled_dynamics_params
      new_gmm_training_state = training_state.gmm_training_state._replace(sample_db_state=new_sample_db_state)
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
          noise_scales = new_noise_scales,
          gmm_training_state=new_gmm_training_state,
          env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, new_key), ()
    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
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
      pdf_values = simul_transitions.target_lnpdf
      
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
      train_length: int = 1,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

    def f(carry, unused_t):
      del unused_t
      ts, es, bs, k = carry
      k, new_key = jax.random.split(k)
      ts, es, bs, metrics, dps, lnpdfs = training_step(ts, es, bs, k)
      return (ts, es, bs, new_key), (metrics, dps, lnpdfs)

    (training_state, env_state, buffer_state, key), (metrics, dynamics_params, target_lnpdfs) = jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, key),
        (),
        length=train_length,
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, env_state, buffer_state, metrics, dynamics_params, target_lnpdfs

  training_epoch_pmap = jax.pmap(functools.partial(training_epoch, train_length=num_training_steps_per_epoch),\
                                  axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, env_state, buffer_state, metrics, samples, target_lnpdfs) = training_epoch_pmap(
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

    return training_state, env_state, buffer_state, metrics, samples, target_lnpdfs  # pytype: disable=bad-return-type  # py311-upgrade
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
  
  print("gmmtd3 OBS SHAPE2", obs_shape)
  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_shape,
      local_devices_to_use=local_devices_to_use,
      gmmtd3_network=gmmtd3_network,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      gmm_init_state=gmm_init_state,
      num_envs=num_envs,
      std_max=std_max,
      std_min=std_min,
  )
  del global_key

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
  if eval_with_training_env:
    eval_dr_low, eval_dr_high = training_dr_range
    eval_env = wrap_for_adv_training(
      eval_env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=functools.partial(randomization_fn, dr_range=training_dr_range),
      param_size=dynamics_param_size,
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
    evaluation_randomization_fn = eval_randomization_fn or randomization_fn
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
  ed = time.time()
  print("setup time", ed-st)
  # # Create and initialize the replay buffer.
  sample_key, local_key = jax.random.split(local_key)
  t = time.time()
  
  if process_id ==0:
    if can_visualize_dr:
      samples = gmmtd3_network.gmm_network.model.sample(_unpmap(training_state.gmm_training_state.model_state.gmm_state), sample_key, 1000)[0]
      log_prob_fn = jax.vmap(functools.partial(gmmtd3_network.gmm_network.model.log_density,\
                                              gmm_state=_unpmap(training_state.gmm_training_state.model_state.gmm_state)))
      model_fig, model_fig_raw = gmm_utils.visualise(
        log_prob_fn,
        dr_range_low,
        dr_range_high,
        samples,
        bijector_log_prob=gmmtd3_network.gmm_network.model.bijector_log_prob
      )
      if use_wandb:
        wandb.log(
                {"model" :wandb.Image(model_fig)},
                step=int(0),
            )
        if model_fig_raw is not None:
            wandb.log(
                  {"model_raw" :wandb.Image(model_fig_raw)},
                  step=int(0),
              )
    metrics = evaluator.run_evaluation(
      _unpmap(
            (training_state.normalizer_params, training_state.policy_params)
      ),
      training_metrics={},

    )
    logging.info(metrics)
    progress_fn(0, metrics)

    print("fig initial step")
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
  if can_visualize_dr:
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
    target_entropy = target_pdfs.mean()
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
  for epoch in range(num_evals_after_init):
    logging.info('step %s', current_step)

    # Optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    target_fig = None
    # if current_step < num_timesteps/5:
    #   (training_state, env_state, buffer_state, training_metrics) = (
    #       training_without_gmm(
    #           training_state, env_state, buffer_state, epoch_keys
    #       )
    #   )
    # else: 
    (training_state, env_state, buffer_state, training_metrics, samples, target_lnpdfs) = (
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
      if can_visualize_dr:
        sample_key, local_key = jax.random.split(local_key)
        samples = gmmtd3_network.gmm_network.model.sample(_unpmap(training_state.gmm_training_state.model_state.gmm_state), sample_key, 1000)[0]
        log_prob_fn = jax.vmap(functools.partial(gmmtd3_network.gmm_network.model.log_density,\
                                              gmm_state=_unpmap(training_state.gmm_training_state.model_state.gmm_state)))
        model_fig, model_fig_raw = gmm_utils.visualise(
          log_prob_fn,
          dr_range_low,
          dr_range_high,
          samples,
          bijector_log_prob=gmmtd3_network.gmm_network.model.bijector_log_prob
        )
        if use_wandb:
          if target_fig is not None:
            wandb.log(
                  {"comparison" :
                   [
                     wandb.Image(model_fig, caption="model"),
                     wandb.Image(target_fig, caption="target"),  
                  ]},
                  step=int(current_step),
              )
          else:
            wandb.log(
                  {"model" :wandb.Image(model_fig)},
                  step=int(current_step),
              )
          if model_fig_raw is not None:
            wandb.log(
                  {"model_raw" :wandb.Image(model_fig_raw)},
                  step=int(current_step),
              )
      metrics = evaluator.run_evaluation(
        _unpmap(
              (training_state.normalizer_params, training_state.policy_params)
        ),
          training_metrics,
      )
      logging.info(metrics)
      #evaluation on current occupancy
      if can_visualize_dr:
        evaluation_key, local_key = jax.random.split(local_key)
        evaluation_key = jax.random.split(evaluation_key, local_devices_to_use)
        target_pdfs = evaluation_on_current_occupancy(
            training_state, env_state, buffer_state, evaluation_key
        )
        x, y = jnp.meshgrid(jnp.linspace(dr_range_low[0], dr_range_high[0], 32),\
                              jnp.linspace(dr_range_low[1], dr_range_high[1], 32))
        target_pdfs = target_pdfs.mean(axis=(0,2))
        target_pdfs = target_pdfs.reshape(x.shape)
        target_entropy = target_pdfs.mean()
        if process_id==0:
          target_fig = plt.figure()
          ctf = plt.contourf(x, y, target_pdfs, levels=20, cmap='viridis')
          cbar = target_fig.colorbar(ctf)
          if use_wandb:
            wandb.log({
              'target_prob on current occupancy with critic' : wandb.Image(target_fig)
            }, step=current_step)
        metrics.update({'target entropy' : target_entropy})
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
  # pmap.assert_is_replicated(training_state)
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (functools.partial(make_policy, deterministic=True), params, metrics)
