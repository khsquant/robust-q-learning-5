from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from mujoco import mjx
from custom_envs import mjx_env
from brax.envs.base import Wrapper, Env, State
from brax.base import System
import functools
import time
import scipy
import numpy as np
from flax import struct
from brax import envs

def build_eval_grid(dr_range_low, dr_range_high, num_points, min_side=4, seed=0):
    """도메인 파라미터 공간(임의 차원 dim)에서 고정된 평가 도메인 집합을 만든다.
       반환: 항상 정확히 (num_points, dim) 배열.
       - side >= min_side  : 축마다 등간격 텐서곱 격자(+부족분은 고정 uniform으로 채움)
       - 그 외(고차원)     : 전부 고정 seed uniform 샘플
    """
    low  = jnp.asarray(dr_range_low,  dtype=jnp.float32)
    high = jnp.asarray(dr_range_high, dtype=jnp.float32)
    dim  = int(low.shape[0])
    key  = jax.random.PRNGKey(seed)

    # 축당 점 개수 side: side**dim <= num_points 가 되도록 (부동소수 오차까지 보정)
    side = int(round(num_points ** (1.0 / dim)))
    while side > 1 and side ** dim > num_points:
        side -= 1

    if side >= min_side:
        axes = [jnp.linspace(low[d], high[d], side) for d in range(dim)]   # 축마다 등간격
        mesh = jnp.meshgrid(*axes, indexing='ij')                          # dim개 텐서
        grid = jnp.stack([m.ravel() for m in mesh], axis=-1)               # (side**dim, dim)
        n_pad = num_points - grid.shape[0]                                 # 남는 자리
        if n_pad > 0:                                                      # 고정 uniform으로 정확히 채움
            pad = jax.random.uniform(key, (n_pad, dim), minval=low, maxval=high)
            grid = jnp.concatenate([grid, pad], axis=0)
        return grid                                                        # (num_points, dim)

    # 고차원: 격자가 성기므로 전부 고정 seed uniform (재현성 보장)
    return jax.random.uniform(key, (num_points, dim), minval=low, maxval=high)


def sample_dynamics_params(
    key: jax.Array,
    num_envs: int,
    dr_range_low: jnp.ndarray,
    dr_range_high: jnp.ndarray,
) -> jnp.ndarray:
  return jax.random.uniform(
      key,
      shape=(num_envs, len(dr_range_low)),
      minval=dr_range_low,
      maxval=dr_range_high,
  )


@struct.dataclass
class EvalMetrics:
  """Dataclass holding evaluation metrics for Brax.

  Attributes:
      episode_metrics: Aggregated episode metrics since the beginning of the
        episode.
      active_episodes: Boolean vector tracking which episodes are not done yet.
      episode_steps: Integer vector tracking the number of steps in the episode.
  """

  episode_metrics: Dict[str, jax.Array]
  active_episodes: jax.Array
  episode_steps: jax.Array


from brax.training.types import Policy, PolicyParams, PRNGKey, Metrics, Transition
class AdvEvalWrapper(Wrapper):
  """Brax env with eval metrics."""

  def reset(self, rng: jax.Array) -> State:
    reset_state = self.env.reset(rng)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(
            jnp.zeros_like, reset_state.metrics
        ),
        active_episodes=jnp.ones_like(reset_state.reward),
        episode_steps=jnp.zeros_like(reset_state.reward),
    )
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state

  def step(self, state: State, action: jax.Array, params: jax.Array) -> State:
    state_metrics = state.info['eval_metrics']
    if not isinstance(state_metrics, EvalMetrics):
      raise ValueError(
          f'Incorrect type for state_metrics: {type(state_metrics)}'
      )
    del state.info['eval_metrics']
    nstate = self.env.step(state, action, params)
    nstate.metrics['reward'] = nstate.reward
    episode_steps = jnp.where(
        state_metrics.active_episodes,
        nstate.info['steps'],
        state_metrics.episode_steps,
    )
    episode_metrics = jax.tree_util.tree_map(
        lambda a, b: a + b * state_metrics.active_episodes,
        state_metrics.episode_metrics,
        nstate.metrics,
    )
    active_episodes = state_metrics.active_episodes * (1 - nstate.done)

    eval_metrics = EvalMetrics(
        episode_metrics=episode_metrics,
        active_episodes=active_episodes,
        episode_steps=episode_steps,
    )
    nstate.info['eval_metrics'] = eval_metrics
    return nstate


class TransitionwithParams(NamedTuple):
  """Transition with additional dynamics parameters."""
  observation: jax.Array
  dynamics_params: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  next_observation: jax.Array
  extras: Dict[str, Any] = {}
def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
  """Collect data."""
  actions, policy_extras = policy(env_state.obs, key)
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
  )
def adv_step(
  env: Env,
  env_state: State,
  dynamics_params: jnp.ndarray,
  policy: Policy,
  key: PRNGKey,
  extra_fields: Sequence[str] = (),
):
  
  actions, policy_extras = policy(env_state.obs, key)

  nstate = env.step(env_state, actions, dynamics_params)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      dynamics_params=dynamics_params,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation= nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
      )
def mpc_step(
  env: Env,
  env_state: State,
  prev_plan : jnp.ndarray,
  policy: Policy,
  key: PRNGKey,
  extra_fields: Sequence[str] = (),
):
  actions, policy_extras = policy(env_state.obs['state'], prev_plan, key)
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, policy_extras['plan'], Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation= nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
      )
def mpc_adv_step(
  env: Env,
  env_state: State,
  prev_plan : jnp.ndarray,
  dynamics_params: jnp.ndarray,
  policy: Policy,
  key: PRNGKey,
  extra_fields: Sequence[str] = (),
):
  actions, policy_extras = policy(env_state.obs['state'], prev_plan, key)
  nstate = env.step(env_state, actions, dynamics_params)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, policy_extras['plan'], TransitionwithParams(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      dynamics_params=dynamics_params,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation= nstate.obs,
      extras={'policy_extras': policy_extras, 'state_extras': state_extras},
      )
def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    use_mpc:bool=False,
    dummy_plan:jnp.ndarray = jnp.zeros(1),
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""
  @jax.jit
  def m(carry, unused_t):
    state, prev_plan, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, current_plan, transition  = mpc_step(
        env, state, prev_plan, policy, current_key, extra_fields=extra_fields
    )
    return (nstate, current_plan, next_key), transition
  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, current_key, extra_fields=extra_fields
    )
    return (nstate, next_key), transition
  if use_mpc:
    (final_state, _, _), data = jax.lax.scan(
        m, (env_state,dummy_plan, key), (), length=unroll_length
    )
  else:
    (final_state, _), data = jax.lax.scan(
        f, (env_state, key), (), length=unroll_length
    )
  return final_state, data


def generate_adv_unroll(
    env: Env,
    env_state: State,
    dynamics_params :jnp.ndarray,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    use_mpc: bool=False, 
    dummy_plan : jnp.ndarray = jnp.zeros(1),
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""
  @jax.jit
  def m(carry, unused_t):
    state, prev_plan, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, current_plan, transition  = mpc_adv_step(
        env, state, prev_plan, dynamics_params, policy, current_key, extra_fields=extra_fields
    )
    return (nstate, current_plan, next_key), transition
  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = adv_step(
        env, state, dynamics_params, policy, current_key, extra_fields=extra_fields
    )
    return (nstate, next_key), transition
  if use_mpc:
    (final_state, _, _), data = jax.lax.scan(
        m, (env_state,dummy_plan, key), (), length=unroll_length
    )
  else:
    (final_state, _), data = jax.lax.scan(
        f, (env_state, key), (), length=unroll_length
    )
  return final_state, data


# TODO: Consider moving this to its own file.
class Evaluator:
  def __init__(
      self,
      eval_env: Env,
      eval_policy_fn: Callable[[PolicyParams], Policy],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
      use_mpc: bool=False,
      dummy_plan:jnp.ndarray = jnp.zeros(1),
  ):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.0

    eval_env = envs.training.EvalWrapper(eval_env)

    def generate_eval_unroll(
        policy_params: PolicyParams, key: PRNGKey
    ) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat,
          use_mpc=use_mpc,
          dummy_plan=dummy_plan,
      )[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll) #jax.jit
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: PolicyParams,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)
    t = time.time()
    eval_state = self._generate_eval_unroll(policy_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    iqm_fn = functools.partial(scipy.stats.trim_mean, proportiontocut=0.25, axis=None) 
    # for fn in [np.mean, np.std, iqm_fn]:
    #   suffix = '_std' if fn == np.std else '_iqm' if fn == iqm_fn else ''
    #   metrics.update({
    #       f'eval/episode_{name}{suffix}': (
    #           fn(value) if aggregate_episodes else value
    #       )
    #       for name, value in eval_metrics.episode_metrics.items()
    #   })
    metrics['eval/episode_reward_mean'] = np.mean(eval_metrics.episode_metrics['reward'])
    metrics['eval/episode_reward_p5'] = np.percentile(eval_metrics.episode_metrics['reward'],5)
    metrics['eval/episode_reward_p10'] = np.percentile(eval_metrics.episode_metrics['reward'],10)
    metrics['eval/episode_reward_p15'] = np.percentile(eval_metrics.episode_metrics['reward'],15)
    metrics['eval/episode_reward_p20'] = np.percentile(eval_metrics.episode_metrics['reward'],20)
    metrics['eval/episode_reward_p25'] = np.percentile(eval_metrics.episode_metrics['reward'],25)
    metrics['eval/episode_reward_p50'] = np.percentile(eval_metrics.episode_metrics['reward'],50)
    metrics['eval/episode_reward_p75'] = np.percentile(eval_metrics.episode_metrics['reward'],75)
    metrics['eval/episode_reward_std'] = np.std(eval_metrics.episode_metrics['reward'])
    metrics['eval/episode_reward_min'] = np.min(eval_metrics.episode_metrics['reward'])
    
    _r = np.sort(np.asarray(eval_metrics.episode_metrics['reward']).ravel())
    metrics['eval/episode_reward_cvar10'] = _r[:max(1, int(_r.shape[0]*0.10))].mean()
    metrics['eval/episode_reward_cvar20'] = _r[:max(1, int(_r.shape[0]*0.20))].mean()
    
    metrics['eval/episode_reward_max'] = np.max(eval_metrics.episode_metrics['reward'])
    metrics['eval/episode_reward_iqm'] = scipy.stats.trim_mean(eval_metrics.episode_metrics['reward'], proportiontocut=0.25, axis=None)
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray

class AdvEvaluator:
  """Class to run evaluations."""

  def __init__(
      self,
      eval_env: Env,
      eval_policy_fn: Callable[[PolicyParams], Policy],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
      dr_range_low: Optional[jnp.ndarray] = None,
      dr_range_high: Optional[jnp.ndarray] = None,
      num_eval_seeds: int = 5,
      use_mpc: bool = False,
      dummy_plan:jnp.ndarray = jnp.zeros(1),
  ):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.0
    self._num_eval_envs = num_eval_envs
    self._dr_range_low = dr_range_low
    self._dr_range_high = dr_range_high
    self._num_eval_seeds = num_eval_seeds
    # AdvEvaluator.__init__ 안, self._num_eval_seeds = ... 근처에 추가
    self._eval_grid = None
    if dr_range_low is not None and dr_range_high is not None:
        self._eval_grid = build_eval_grid(dr_range_low, dr_range_high, num_eval_envs)

    eval_env = AdvEvalWrapper(eval_env)

    def generate_eval_unroll(
        policy_params: PolicyParams, eval_dynamics_params, key: PRNGKey
    ) -> State:
      reset_keys = jax.random.split(key, len(eval_dynamics_params))
      eval_first_state = eval_env.reset(reset_keys)
      return generate_adv_unroll(
          eval_env,
          eval_first_state,
          eval_dynamics_params,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat,
          use_mpc=use_mpc,
          dummy_plan=dummy_plan,
      )[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll) #jax.jit
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      policy_params: PolicyParams,
      dynamics_params: Optional[jnp.ndarray] = None,
      training_metrics: Optional[Metrics] = None,
      aggregate_episodes: bool = True,
      num_eval_seeds: Optional[int] = None,
      return_reward_array: bool = False,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    if training_metrics is None and isinstance(dynamics_params, dict):
      training_metrics = dynamics_params
      dynamics_params = None
    if training_metrics is None:
      training_metrics = {}
    if num_eval_seeds is None:
      num_eval_seeds = self._num_eval_seeds


    if dynamics_params is None:
      self._key, params_key, unroll_key = jax.random.split(self._key, 3)
      if self._eval_grid is not None:
        dynamics_params = self._eval_grid                       # 고정 격자 우선
      elif self._dr_range_low is not None and self._dr_range_high is not None:
        dynamics_params = sample_dynamics_params(               # 격자 없으면 기존 랜덤
            params_key,
            self._num_eval_envs,
            self._dr_range_low,
            self._dr_range_high,
        )
      else:
        raise ValueError(
            'AdvEvaluator needs eval_grid or dr_range_low/high '
            'when dynamics_params are not provided.'
        )
    else:
      self._key, unroll_key = jax.random.split(self._key)

    unroll_keys = jax.random.split(unroll_key, num_eval_seeds)
    t = time.time()
    def f(carry, key):
      eval_state = self._generate_eval_unroll(policy_params, dynamics_params, key)
      eval_metrics = eval_state.info['eval_metrics']
      return carry, eval_metrics
    _, eval_metrics = jax.lax.scan(f, None, unroll_keys)
    # eval_state = self._generate_eval_unroll(policy_params, dynamics_params, unroll_key)
    # eval_metrics = eval_state.info['eval_metrics']
    eval_metrics = jax.tree_util.tree_map(lambda x : x.mean(axis=0), eval_metrics)
    reward_2d = eval_metrics.episode_metrics['reward']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    iqm_fn = functools.partial(scipy.stats.trim_mean, proportiontocut=0.25, axis=None) 
    # for fn in [np.mean, np.std, iqm_fn]:
    #   suffix = '_std' if fn == np.std else '_iqm' if fn == iqm_fn else ''
    #   metrics.update({
    #       f'eval/episode_{name}{suffix}': (
    #           fn(value) if aggregate_episodes else value
    #       )
    #       for name, value in eval_metrics.episode_metrics.items()
    #   })
    metrics['eval/episode_reward_mean'] = np.mean(eval_metrics.episode_metrics['reward'])
    metrics['eval/episode_reward_p5'] = np.percentile(eval_metrics.episode_metrics['reward'],5)
    metrics['eval/episode_reward_p10'] = np.percentile(eval_metrics.episode_metrics['reward'],10)
    metrics['eval/episode_reward_p15'] = np.percentile(eval_metrics.episode_metrics['reward'],15)
    metrics['eval/episode_reward_p20'] = np.percentile(eval_metrics.episode_metrics['reward'],20)
    metrics['eval/episode_reward_p25'] = np.percentile(eval_metrics.episode_metrics['reward'],25)
    metrics['eval/episode_reward_p50'] = np.percentile(eval_metrics.episode_metrics['reward'],50)
    metrics['eval/episode_reward_p75'] = np.percentile(eval_metrics.episode_metrics['reward'],75)
    metrics['eval/episode_reward_std'] = np.std(eval_metrics.episode_metrics['reward'])
    metrics['eval/episode_reward_min'] = np.min(eval_metrics.episode_metrics['reward'])

    _r = np.sort(np.asarray(eval_metrics.episode_metrics['reward']).ravel())
    metrics['eval/episode_reward_cvar10'] = _r[:max(1, int(_r.shape[0]*0.10))].mean()
    metrics['eval/episode_reward_cvar20'] = _r[:max(1, int(_r.shape[0]*0.20))].mean()
    
    metrics['eval/episode_reward_max'] = np.max(eval_metrics.episode_metrics['reward'])
    metrics['eval/episode_reward_iqm'] = scipy.stats.trim_mean(eval_metrics.episode_metrics['reward'], proportiontocut=0.25, axis=None)
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    if return_reward_array:
      return metrics, reward_2d  # pytype: disable=bad-return-type  # jax-ndarray
    return metrics  # pytype: disable=bad-return-type  # jax-ndarray
