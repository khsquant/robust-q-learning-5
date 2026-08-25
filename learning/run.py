import functools
import hashlib
import inspect
import json
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _path in (_REPO_ROOT, _THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import hydra
import jax
from ml_collections import config_dict
import numpy as np
import wandb
from mujoco import mjx
from omegaconf import OmegaConf

try:
    from agents.bridgetd3 import networks as bridgetd3_networks
    from agents.bridgetd3 import train as bridgetd3
except ImportError:
    bridgetd3_networks = None
    bridgetd3 = None

try:
    from agents.gmmtd3 import networks as gmmtd3_networks
    from agents.gmmtd3 import train as gmmtd3
    _GMM_TD3_IMPORT_ERROR = None
except ImportError as exc:
    gmmtd3_networks = None
    gmmtd3 = None
    _GMM_TD3_IMPORT_ERROR = exc
from agents.m2td3 import networks as m2td3_networks
from agents.m2td3 import train as m2td3
from agents.sac import networks as sac_networks
from agents.sac import train as sac

from agents.gmmsac import networks as gmmsac_networks # 수정됨
from agents.gmmsac import train as gmmsac

from agents.tcrmdp import networks as tcrmdp_networks
from agents.tcrmdp import train as tcrmdp
from agents.td3 import networks as td3_networks
from agents.td3 import train as td3
from agents.wdsac import networks as wdsac_networks
from agents.wdsac import train as wdsac
from agents.wdtd3 import networks as wdtd3_networks
from agents.wdtd3 import train as wdtd3
from agents.gmmtd3 import networks as gmmtd3_networks
from agents.gmmtd3 import train as gmmtd3
from custom_envs import dm_control_suite, locomotion, manipulation, mjx_env, registry
from helper import make_dir, parse_cfg
from learning.configs.dm_control_training_config import (
    brax_sac_config,
    brax_tcrmdp_config,
    brax_td3_config,
    brax_wdsac_config,
)
from learning.configs.locomotion_training_config import (
    locomotion_sac_config,
    locomotion_tcrmdp_config,
    locomotion_td3_config,
)
from learning.configs.manipulation_training_config import (
    manipulation_tcrmdp_config,
    manipulation_td3_config,
)
from learning.module.wrapper.wrapper import Wrapper
from utils import save_configs_to_wandb_and_local


CAMERAS = {
    "AcrobotSwingup": "fixed",
    "AcrobotSwingupSparse": "fixed",
    "BallInCup": "cam0",
    "CartpoleBalance": "fixed",
    "CartpoleBalanceSparse": "fixed",
    "CartpoleSwingup": "fixed",
    "CartpoleSwingupSparse": "fixed",
    "CheetahRun": "side",
    "FingerSpin": "cam0",
    "FingerTurnEasy": "cam0",
    "FingerTurnHard": "cam0",
    "FishSwim": "fixed_top",
    "HopperHop": "cam0",
    "HopperStand": "cam0",
    "HumanoidStand": "side",
    "HumanoidWalk": "side",
    "HumanoidRun": "side",
    "PendulumSwingup": "fixed",
    "PointMass": "cam0",
    "ReacherEasy": "fixed",
    "ReacherHard": "fixed",
    "SwimmerSwimmer6": "tracking1",
    "WalkerRun": "side",
    "WalkerWalk": "side",
    "WalkerStand": "side",
    "Go1Handstand": "side",
    "Go1JoystickRoughTerrain": "track",
    "G1InplaceGaitTracking": "track",
    "G1JoystickGaitTracking": "track",
    "T1JoystickFlatTerrain": "track",
    "T1JoystickRoughTerrain": "track",
    "LeapCubeRotateZAxis": "side",
    "LeapCubeReorient": "side",
}

_WANDB_GROUP_LIMIT = 120
_WANDB_GROUP_HASH_LENGTH = 8
_TC_WANDB_POLICIES = frozenset({"tc_bridgetd3", "tc_gmmtd3"})

def _limit_wandb_group(group: str) -> str:
    """Keep W&B group names within the API limit while retaining uniqueness."""
    group = str(group)
    if len(group) <= _WANDB_GROUP_LIMIT:
        return group

    digest = hashlib.sha1(group.encode("utf-8")).hexdigest()[
        :_WANDB_GROUP_HASH_LENGTH
    ]
    suffix = f".{digest}"
    limited_group = group[: _WANDB_GROUP_LIMIT - len(suffix)] + suffix
    print(
        "Truncated wandb_group from "
        f"{len(group)} to {len(limited_group)} chars: {limited_group}"
    )
    return limited_group


def _tc_prefixed(name: str) -> str:
    name = str(name)
    return name if name.startswith("tc_") else f"tc_{name}"


def _uses_tc_wandb_prefix(policy: str) -> bool:
    return policy in _TC_WANDB_POLICIES


def _resolve_impl(requested_impl: str) -> str:
    """Maps legacy impl names and falls back from Warp when CUDA is unavailable."""
    if requested_impl == "mjx":
        return "jax"
    if requested_impl != "warp":
        return requested_impl

    has_gpu_backend = any(device.platform == "gpu" for device in jax.devices())
    if has_gpu_backend:
        return requested_impl

    print("Requested impl='warp' without a GPU backend; falling back to impl='jax'.")
    return "jax"


class BraxDomainRandomizationWrapper(Wrapper):
    """Brax wrapper for domain randomized evaluation."""

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    ):
        super().__init__(env)
        self._mjx_model, self._in_axes = randomization_fn(self.env.mjx_model)
        self.env.unwrapped._mjx_model = self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return self.env.reset(rng)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self.env.step(state, action)


def _to_jsonable_metric(value):
    try:
        value = jax.device_get(value)
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    try:
        array_value = np.asarray(value)
        if array_value.shape == ():
            return array_value.item()
        return array_value.tolist()
    except Exception:
        return str(value)


def _append_metrics_jsonl(num_steps, metrics):
    metrics_path = os.environ.get("METRICS_JSONL")
    if not metrics_path:
        return
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    row = {
        "num_steps": int(num_steps),
        "num_update_steps": int(num_steps // 8),
    }
    for key, value in metrics.items():
        row[key] = _to_jsonable_metric(value)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def progress_fn(num_steps, metrics, use_wandb=True):
    if use_wandb:
        wandb.log(metrics, step=num_steps)
    _append_metrics_jsonl(num_steps, metrics)
    print("-------------------------------------------------------------------")
    print(f"num_steps: {num_steps}")
    print(f"num_update_steps: {num_steps // 8}")
    for key, value in metrics.items():
        print(f" {key} :  {value}")
    print("-------------------------------------------------------------------")


def _maybe_override_config(params, cfg):
    for param in params.keys():
        if param in cfg and not OmegaConf.is_missing(cfg, param):
            value = getattr(cfg, param)
            if value is not None:
                if OmegaConf.is_config(value):
                    value = OmegaConf.to_container(value, resolve=True)
                if (
                    isinstance(params[param], config_dict.ConfigDict)
                    and isinstance(value, dict)
                ):
                    value = config_dict.ConfigDict(value)
                params[param] = value


def _filter_kwargs(callable_fn, kwargs):
    signature = inspect.signature(callable_fn)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return dict(kwargs)
    return {
        key: value
        for key, value in dict(kwargs).items()
        if key in signature.parameters
    }


def _init_wandb(cfg, name: str):
    if cfg.use_wandb:
        if cfg.wandb_group:
            group = cfg.wandb_group
        else:
            group_parts =  [name]
            if cfg.wandb_group_prefix:
                group_parts.insert(0, cfg.wandb_group_prefix)
            group = ".".join(str(part) for part in group_parts if part)
        if _uses_tc_wandb_prefix(cfg.policy):
            name = _tc_prefixed(name)
            group = _tc_prefixed(group)
        group = _limit_wandb_group(group)
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config["wandb_group"] = group
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=name + cfg.comment,
            group=group,
            job_type=cfg.policy,
            tags=[
                f"task:{cfg.task}",
                f"policy:{cfg.policy}",
                f"seed:{cfg.seed}",
                f"exp_name:{cfg.exp_name}",
            ],
            dir=make_dir(cfg.work_dir),
            config=wandb_config,
        )
        wandb.config.update({"env_name": cfg.task, "wandb_group": group})


def _sac_config(task: str):
    if task in dm_control_suite._envs:
        return brax_sac_config(task)
    if task in locomotion._envs:
        return locomotion_sac_config(task)
    raise ValueError(f"SAC config is not defined for task {task}.")


def _td3_config(task: str):
    if task in dm_control_suite._envs:
        return brax_td3_config(task)
    if task in locomotion._envs:
        return locomotion_td3_config(task)
    if task in manipulation._envs:
        return manipulation_td3_config(task)
    raise ValueError(f"TD3 config is not defined for task {task}.")


def _tcrmdp_config(task: str):
    if task in dm_control_suite._envs:
        return brax_tcrmdp_config(task)
    if task in locomotion._envs:
        return locomotion_tcrmdp_config(task)
    if task in manipulation._envs:
        return manipulation_tcrmdp_config(task)
    raise ValueError(f"TCRMDP config is not defined for task {task}.")


def _adv_randomizer(task: str, randomization_fn):
    if randomization_fn is None:
        return None
    return registry.get_domain_randomizer_eval(task)


def _cfg_flag(cfg, name: str, default: bool = False) -> bool:
    if name in cfg and not OmegaConf.is_missing(cfg, name):
        value = getattr(cfg, name)
        if value is None:
            return default
        return bool(value)
    return default


def _train_bridgetd3_variant(
    cfg,
    randomization_fn,
    env,
    eval_env=None,
    *,
    use_tc: bool = False,
):
    del eval_env
    if bridgetd3 is None or bridgetd3_networks is None:
        raise ImportError(
            "bridgetd3 dependencies are not available in this environment."
        )
    bridgetd3_params = _td3_config(cfg.task)
    bridgetd3_params.bridge_alpha = 1.0
    bridgetd3_params.bridge_auto_alpha = True
    bridgetd3_params.bridge_target_kinetic_coef = 2.5
    bridgetd3_params.bridge_init_log_alpha = None
    bridgetd3_params.bridge_alpha_lr = None
    bridgetd3_params.adversary_learning_rate = None
    bridgetd3_params.dr_augmented_critic = True
    bridgetd3_params.use_tc = use_tc
    bridgetd3_params.radius = 0.001
    _maybe_override_config(bridgetd3_params, cfg)

    wandb_name = (
        f"{cfg.task}.{cfg.policy}.{cfg.seed}"
        f".alpha={bridgetd3_params.bridge_alpha}.beta={cfg.beta}.radius={cfg.radius}"
    )
    _init_wandb(cfg, wandb_name)

    bridgetd3_training_params = dict(bridgetd3_params)
    network_factory = bridgetd3_networks.make_bridgetd3_networks
    if "network_factory" in bridgetd3_params:
        del bridgetd3_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            bridgetd3_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            bridgetd3_networks.make_bridgetd3_networks,
            **_filter_kwargs(
                bridgetd3_networks.make_bridgetd3_networks,
                bridgetd3_params.network_factory,
            ),
        )
    bridgetd3_training_params = _filter_kwargs(
        bridgetd3.train, bridgetd3_training_params
    )

    train_fn = functools.partial(
        bridgetd3.train,
        **dict(bridgetd3_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=_adv_randomizer(cfg.task, randomization_fn),
        eval_randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
        use_wandb=cfg.use_wandb,
    )
    return train_fn(environment=env)


def _train_gmmtd3_variant(
    cfg,
    randomization_fn,
    env,
    eval_env=None,
    *,
    use_tc: bool = False,
):
    del eval_env
    if gmmtd3 is None or gmmtd3_networks is None:
        raise ImportError(
            "gmmtd3 dependencies are not available in this environment."
        ) from _GMM_TD3_IMPORT_ERROR
    gmmtd3_params = _td3_config(cfg.task)
    gmmtd3_params.use_tc = use_tc
    gmmtd3_params.radius = 0.001
    gmmtd3_params.num_eval_envs = 16384 if cfg.task == "WalkerWalk" else 4096
    _maybe_override_config(gmmtd3_params, cfg)

    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.beta={cfg.beta}.radius={cfg.radius}"
    _init_wandb(cfg, wandb_name)

    gmmtd3_training_params = dict(gmmtd3_params)
    if "network_factory" in gmmtd3_params:
        if not cfg.asymmetric_critic:
            gmmtd3_params.network_factory.value_obs_key = "state"
        del gmmtd3_training_params["network_factory"]
        network_factory = functools.partial(
            gmmtd3_networks.make_gmmtd3_networks,
            **gmmtd3_params.network_factory,
        )
    else:
        network_factory = gmmtd3_networks.make_gmmtd3_networks

    train_fn = functools.partial(
        gmmtd3.train,
        **dict(gmmtd3_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        eval_randomization_fn=randomization_fn,
        randomization_fn=registry.get_domain_randomizer_eval(cfg.task),
        use_wandb=cfg.use_wandb,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
        eval_with_training_env=cfg.eval_with_training_env,
        value_obs_key=gmmtd3_params.network_factory.value_obs_key,
        dr_augmented_critic=_cfg_flag(cfg, "dr_augmented_critic"),
        beta=cfg.beta,
    )
    return train_fn(environment=env)


def train_sac(cfg, randomization_fn, env, eval_env=None):
    sac_params = _sac_config(cfg.task)
    sac_params.dr_augmented_critic = _cfg_flag(cfg, "dr_augmented_critic")
    sac_params.num_eval_envs = 16384 if cfg.task == "WalkerWalk" else 4096
    _maybe_override_config(sac_params, cfg)
    sac_training_params = dict(sac_params)
    wandb_name = (
        f"{cfg.task}.{cfg.policy}.{cfg.seed}"
    )
    _init_wandb(cfg, wandb_name)

    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            sac_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            sac_networks.make_simba_sac_networks
            if cfg.simba
            else sac_networks.make_sac_networks,
            **sac_params.network_factory,
        )
    else:
        network_factory = (
            sac_networks.make_simba_sac_networks
            if cfg.simba
            else sac_networks.make_sac_networks
        )

    train_fn = functools.partial(
        sac.train,
        **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=_adv_randomizer(cfg.task, randomization_fn),
        eval_randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
    )
    return train_fn(environment=env)


def train_wdsac(cfg, randomization_fn, env, eval_env=None):
    if randomization_fn is None:
        raise ValueError("WDSAC requires randomization=true.")
    if cfg.task in dm_control_suite._envs:
        wdsac_params = brax_wdsac_config(cfg.task)
    elif cfg.task in locomotion._envs:
        wdsac_params = locomotion_sac_config(cfg.task)
    else:
        raise ValueError(f"WDSAC config is not defined for task {cfg.task}.")

    wdsac_params.n_nominals = 10
    wdsac_params.delta = 0.1
    wdsac_params.lambda_update_steps = 100
    wdsac_params.single_lambda = False
    wdsac_params.lmbda_lr = 3e-4
    wdsac_params.init_lmbda = 1.0
    _maybe_override_config(wdsac_params, cfg)

    wandb_name = (
        f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.delta={wdsac_params.delta}"
        f".nominals={wdsac_params.n_nominals}.single_lambda={wdsac_params.single_lambda}"
        f".asym={cfg.asymmetric_critic}.distance_type={wdsac_params.distance_type}"
        f".length={wdsac_params.lambda_update_steps}.lmbda_lr={wdsac_params.lmbda_lr}"
        f".init_lmbda={wdsac_params.init_lmbda}"
    )
    _init_wandb(cfg, wandb_name)

    wdsac_training_params = dict(wdsac_params)
    network_factory = wdsac_networks.make_wdsac_networks
    if "network_factory" in wdsac_params:
        if not cfg.asymmetric_critic:
            wdsac_params.network_factory.value_obs_key = "state"
        del wdsac_training_params["network_factory"]
        network_factory = functools.partial(
            wdsac_networks.make_wdsac_networks,
            **wdsac_params.network_factory,
        )

    train_fn = functools.partial(
        wdsac.train,
        **dict(wdsac_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
    )
    return train_fn(
        environment=env,
        eval_env=eval_env,
    )

def train_gmmsac(cfg, randomization_fn, env, eval_env=None): # 수정됨
    gmmsac_params = _sac_config(cfg.task)
    gmmsac_params.dr_augmented_critic = _cfg_flag(cfg, "dr_augmented_critic")
    gmmsac_params.num_eval_envs = 16384 if cfg.task == "WalkerWalk" else 4096
    _maybe_override_config(gmmsac_params, cfg)
    gmmsac_training_params = dict(gmmsac_params)
    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.beta={cfg.beta}"
    _init_wandb(cfg, wandb_name)

    if "network_factory" in gmmsac_params:
        del gmmsac_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            gmmsac_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            gmmsac_networks.make_gmmsac_networks,
            **gmmsac_params.network_factory,
        )
    else:
        network_factory = gmmsac_networks.make_gmmsac_networks

    train_fn = functools.partial(
        gmmsac.train,
        **dict(gmmsac_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=_adv_randomizer(cfg.task, randomization_fn),
        eval_randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
        beta=cfg.beta,
    )
    return train_fn(environment=env)

def train_td3(cfg, randomization_fn, env, eval_env=None):
    td3_params = _td3_config(cfg.task)
    td3_params.dr_augmented_critic = _cfg_flag(cfg, "dr_augmented_critic")
    td3_params.num_eval_envs = 16384 if cfg.task == "WalkerWalk" else 4096
    _maybe_override_config(td3_params, cfg)
    td3_training_params = dict(td3_params)
    wandb_name = (
        f"{cfg.task}.{cfg.policy}.{cfg.seed}"
    )
    _init_wandb(cfg, wandb_name)

    network_factory = td3_networks.make_td3_networks
    if "network_factory" in td3_params:
        del td3_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            td3_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            td3_networks.make_td3_networks,
            **td3_params.network_factory,
        )

    train_fn = functools.partial(
        td3.train,
        **dict(td3_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=_adv_randomizer(cfg.task, randomization_fn),
        eval_randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
        use_wandb=cfg.use_wandb,
    )
    return train_fn(environment=env)


def train_m2td3(cfg, randomization_fn, env, eval_env=None):
    m2td3_params = _td3_config(cfg.task)
    m2td3_params.omega_distance_threshold = 0.1
    m2td3_params.omega_noise_rate = 0.2
    m2td3_params.omega_std = 1.0
    m2td3_params.omega_clip = 0.5
    m2td3_params.num_omegas = 5
    m2td3_params.omega_lr = None
    m2td3_params.omega_min_probability = 5e-2
    m2td3_params.omega_prob_update_rate = None
    m2td3_params.omega_restart_distance = True
    m2td3_params.omega_restart_probability = True
    m2td3_params.dr_augmented_critic = True
    m2td3_params.num_eval_envs = 16384 if cfg.task == "WalkerWalk" else 4096
    _maybe_override_config(m2td3_params, cfg)

    wandb_name = (
        f"{cfg.task}.{cfg.policy}.{cfg.seed}"
        f".dist={m2td3_params.omega_distance_threshold}"
    )
    _init_wandb(cfg, wandb_name)

    m2td3_training_params = dict(m2td3_params)
    network_factory = m2td3_networks.make_m2td3_networks
    if "network_factory" in m2td3_params:
        del m2td3_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            m2td3_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            m2td3_networks.make_m2td3_networks,
            **_filter_kwargs(
                m2td3_networks.make_m2td3_networks,
                m2td3_params.network_factory,
            ),
        )
    m2td3_training_params = _filter_kwargs(m2td3.train, m2td3_training_params)

    train_fn = functools.partial(
        m2td3.train,
        **dict(m2td3_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=_adv_randomizer(cfg.task, randomization_fn),
        eval_randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
    )
    return train_fn(environment=env)


def train_bridgetd3(cfg, randomization_fn, env, eval_env=None):
    return _train_bridgetd3_variant(
        cfg,
        randomization_fn,
        env,
        eval_env=eval_env,
        use_tc=False,
    )


def train_tc_bridgetd3(cfg, randomization_fn, env, eval_env=None):
    return _train_bridgetd3_variant(
        cfg,
        randomization_fn,
        env,
        eval_env=eval_env,
        use_tc=True,
    )


def train_gmmtd3(cfg, randomization_fn, env, eval_env=None):
    return _train_gmmtd3_variant(
        cfg,
        randomization_fn,
        env,
        eval_env=eval_env,
        use_tc=False,
    )


def train_tc_gmmtd3(cfg, randomization_fn, env, eval_env=None):
    return _train_gmmtd3_variant(
        cfg,
        randomization_fn,
        env,
        eval_env=eval_env,
        use_tc=True,
    )


def train_tcrmdp_algorithm(cfg, randomization_fn, env, algorithm: str, eval_env=None):
    if randomization_fn is None:
        raise ValueError(f"{algorithm} requires randomization=true.")
    params = _tcrmdp_config(cfg.task)
    params.omniscient_adversary = cfg.omniscient_adversary
    params.asymmetric_critic = cfg.asymmetric_critic
    params.dr_augmented_critic = _cfg_flag(cfg, "dr_augmented_critic")
    _maybe_override_config(params, cfg)

    if algorithm == tcrmdp_networks.RARL:
        wandb_name = (
            f"{cfg.task}.{algorithm}.seed={cfg.seed}"
            f".omniscient={params.omniscient_adversary}"
        )
    else:
        wandb_name = (
            f"{cfg.task}.{algorithm}.seed={cfg.seed}.radius={params.radius}"
        )
    _init_wandb(cfg, wandb_name)

    training_params = dict(params)
    network_factory = functools.partial(
        tcrmdp_networks.make_tcrmdp_networks,
        **training_params.pop("network_factory"),
    )

    train_fn = functools.partial(
        tcrmdp.train,
        **training_params,
        algorithm=algorithm,
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=_adv_randomizer(cfg.task, randomization_fn),
        eval_randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
    )
    return train_fn(environment=env)


def train_rarl(cfg, randomization_fn, env, eval_env=None):
    return train_tcrmdp_algorithm(
        cfg,
        randomization_fn,
        env,
        tcrmdp_networks.RARL,
        eval_env=eval_env,
    )


def train_vanilla_tc_m2td3(cfg, randomization_fn, env, eval_env=None):
    return train_tcrmdp_algorithm(
        cfg,
        randomization_fn,
        env,
        tcrmdp_networks.VANILLA_TC_M2TD3,
        eval_env=eval_env,
    )


def train_tc_rarl(cfg, randomization_fn, env, eval_env=None):
    return train_tcrmdp_algorithm(
        cfg,
        randomization_fn,
        env,
        tcrmdp_networks.TC_RARL,
        eval_env=eval_env,
    )


def train_tc_m2td3(cfg, randomization_fn, env, eval_env=None):
    return train_tcrmdp_algorithm(
        cfg,
        randomization_fn,
        env,
        tcrmdp_networks.TC_M2TD3,
        eval_env=eval_env,
    )


def train_wdtd3(cfg, randomization_fn, env, eval_env=None):
    wdtd3_params = _td3_config(cfg.task)
    wdtd3_params.n_nominals = 10
    wdtd3_params.delta = 0.01
    wdtd3_params.lambda_update_steps = 100
    wdtd3_params.single_lambda = False
    wdtd3_params.distance_type = "wass"
    wdtd3_params.lmbda_lr = 3e-4
    wdtd3_params.init_lmbda = 0.0
    _maybe_override_config(wdtd3_params, cfg)

    wandb_name = (
        f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.delta={wdtd3_params.delta}"
        f".nominals={wdtd3_params.n_nominals}.single_lambda={wdtd3_params.single_lambda}"
        f".asym={cfg.asymmetric_critic}.distance_type={wdtd3_params.distance_type}"
        f".length={wdtd3_params.lambda_update_steps}.lmbda_lr={wdtd3_params.lmbda_lr}"
        f".init_lmbda={wdtd3_params.init_lmbda}"
    )
    _init_wandb(cfg, wandb_name)

    wdtd3_training_params = dict(wdtd3_params)
    network_factory = wdtd3_networks.make_wdtd3_networks
    if "network_factory" in wdtd3_params:
        if not cfg.asymmetric_critic:
            wdtd3_params.network_factory.value_obs_key = "state"
        del wdtd3_training_params["network_factory"]
        network_factory = functools.partial(
            wdtd3_networks.make_wdtd3_networks,
            **wdtd3_params.network_factory,
        )

    train_fn = functools.partial(
        wdtd3.train,
        **dict(wdtd3_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        seed=cfg.seed,
        randomization_fn=randomization_fn,
    )
    return train_fn(environment=env)


TRAINERS = {
    "sac": train_sac,
    "wdsac": train_wdsac,
    "td3": train_td3,
    "m2td3": train_m2td3,
    "bridgetd3": train_bridgetd3,
    "tc_bridgetd3": train_tc_bridgetd3,
    "gmmtd3": train_gmmtd3,
    "gmmsac": train_gmmsac, # 수정됨
    "tc_gmmtd3": train_tc_gmmtd3,
    "rarl": train_rarl,
    "vanilla_tc_m2td3": train_vanilla_tc_m2td3,
    "tc_rarl": train_tc_rarl,
    "tc_m2td3": train_tc_m2td3,
    "wdtd3": train_wdtd3,
}


@hydra.main(config_name="config", config_path=".", version_base=None)
def train(cfg):
    cfg = parse_cfg(cfg)
    print("cfg :", cfg)
    np.set_printoptions(precision=3, suppress=True, linewidth=100)

    rng = jax.random.PRNGKey(cfg.seed)
    cfg_dir = make_dir(cfg.work_dir / "cfg")
    shutil.copy("config.yaml", os.path.join(cfg_dir, "config.yaml"))

    env_cfg = registry.get_default_config(cfg.task)
    env_cfg["impl"] = _resolve_impl(cfg.impl)
    fast_td3_tuned_reward_tasks = {
        "G1JoystickFlatTerrain",
        "G1JoystickRoughTerrain",
        "T1JoystickFlatTerrain",
        "T1JoystickRoughTerrain",
    }
    if cfg.use_tuned_reward and cfg.task in fast_td3_tuned_reward_tasks:
        env_cfg.reward_config.scales.energy = -5e-5
        env_cfg.reward_config.scales.action_rate = -1e-1
        env_cfg.reward_config.scales.torques = -1e-3
        env_cfg.reward_config.scales.pose = -1.0
        env_cfg.reward_config.scales.tracking_ang_vel = 1.25
        env_cfg.reward_config.scales.tracking_lin_vel = 1.25
        env_cfg.reward_config.scales.feet_phase = 1.0
        env_cfg.reward_config.scales.ang_vel_xy = -0.3
        env_cfg.reward_config.scales.orientation = -5.0

    env = registry.load(cfg.task, config=env_cfg)
    randomization_fn = (
        registry.get_domain_randomizer_eval(cfg.task) if cfg.randomization else None
    )
    print("randomization_fn:", randomization_fn)

    if cfg.policy not in TRAINERS:
        supported = ", ".join(sorted(TRAINERS))
        raise ValueError(f"Unsupported q-learning policy '{cfg.policy}'. Use one of: {supported}.")

    make_inference_fn, params, metrics = TRAINERS[cfg.policy](
        cfg, randomization_fn, env
    )

    save_dir = make_dir(cfg.work_dir / "models")
    print(f"Saving parameters to {save_dir}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(save_dir, f"{cfg.policy}_params_{timestamp}.pkl"), "wb") as f:
        pickle.dump(params, f)
    with open(os.path.join(save_dir, f"{cfg.policy}_params_latest.pkl"), "wb") as f:
        pickle.dump(params, f)

    save_configs_to_wandb_and_local(cfg, cfg.work_dir)
    print("eval_randomization", cfg.eval_randomization)
    if cfg.eval_randomization:
        eval_rng, rng = jax.random.split(rng)
        randomizer_eval = registry.get_domain_randomizer_eval(cfg.task)
        randomizer_eval = functools.partial(
            randomizer_eval,
            rng=eval_rng,
            dr_range=env.dr_range,
        )
        eval_env = BraxDomainRandomizationWrapper(
            registry.load(cfg.task, config=env_cfg),
            randomization_fn=randomizer_eval,
        )
    else:
        eval_env = registry.load(cfg.task, config=env_cfg)

    if cfg.save_video and cfg.use_wandb:
        n_episodes = 10
        jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        reward_list = []
        rollout = []
        rng, eval_rng = jax.random.split(rng)
        rngs = jax.random.split(eval_rng, n_episodes)
        for i in range(n_episodes):
            state = jit_reset(rngs[i])
            rollout = [state]
            rewards = 0
            for _ in range(env_cfg.episode_length):
                act_rng, rng = jax.random.split(rng)
                action, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, action)
                rollout.append(state)
                rewards += state.reward
            reward_list.append(rewards)

        frames = eval_env.render(rollout, camera=CAMERAS[cfg.task])
        frames = np.stack(frames).transpose(0, 3, 1, 2)
        wandb.log(
            {
                "eval/video": wandb.Video(frames, fps=1.0 / eval_env.dt, format="mp4"),
                "eval/rewards": np.mean(reward_list),
            }
        )
    return make_inference_fn, params, metrics


if __name__ == "__main__":
    train()
