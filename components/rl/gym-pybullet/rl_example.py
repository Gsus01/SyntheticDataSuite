"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

This version has been modified to:
- Read experiment configuration from a JSON file (e.g. variables.json)
- Run training and evaluation
- Emit a JSON file with metrics for comparison across runs

"""

import os
# Headless-friendly defaults for container execution
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import time
import json
from datetime import datetime
import argparse
from typing import Optional

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# Standard directories for containerized execution
DEFAULT_INPUT_DIR = "/data/inputs"
DEFAULT_OUTPUT_DIR = "/data/outputs"
DEFAULT_CONFIG_DIR = "/data/config"
DEFAULT_CONFIG_FILE = "variables.json"
DEFAULT_METRICS_FILE = "metrics.json"

# --------------------------
# Config por defecto
# --------------------------

DEFAULT_CONFIG = {
    # Entorno
    "multiagent": False,
    "num_drones": 2,
    "observation_type": "kin",          # "kin" o "rgb"
    "action_type": "one_d_rpm",         # "rpm", "pid", "vel", "one_d_rpm", ...

    # Hiperparámetros PPO
    "total_timesteps": int(1e6),
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.0,
    "seed": 0,
    "device": "auto",                   # "cpu", "cuda" o "auto"

    # Evaluación
    "eval_freq": 10000,
    "n_eval_episodes": 5,
    "n_test_episodes": 10,
    "target_reward": None,              # Si es None, se calcula por defecto según action_type / multiagent
    "use_reward_threshold_stop": True,

    # Entorno de ejecución
    "gui": False,
    "record_video": False,
    "plot": False,
    "colab": False,
    "output_folder": "results",

    # Objetivo de hover para métricas de posición (solo single-agent + KIN)
    "pos_target": [0.0, 0.0, 1.0]
}


def load_config(config_path: str) -> dict:
    """Carga el JSON de configuración y lo mezcla con DEFAULT_CONFIG."""
    cfg = DEFAULT_CONFIG.copy()
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config JSON not found: {config_path}")
    with open(config_path, "r") as f:
        user_cfg = json.load(f)
    cfg.update(user_cfg)
    return cfg


def resolve_output_folder(config_value: str, output_dir: Path) -> str:
    """Resolve the output folder path honoring the standard output dir."""
    if not config_value:
        return str(output_dir)
    cfg_path = Path(config_value).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = output_dir / cfg_path
    return str(cfg_path)


def resolve_with_base(base_dir: Path, override: Optional[str], default_name: str) -> Path:
    """Return absolute path using base_dir for relative overrides."""
    candidate = Path(default_name if not override else override).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def safe_mean_std(values):
    """Devuelve (mean, std) ignorando None. Si no hay valores válidos, (None, None)."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None, None
    arr = np.array(xs, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def run(config: dict, metrics_output_path: str):

    script_start = datetime.now()

    # --------------------------
    # Preparar carpetas y run_id
    # --------------------------
    run_id = 'save-' + script_start.strftime("%m.%d.%Y_%H.%M.%S")
    output_folder = config.get("output_folder", "results")
    run_folder = os.path.join(output_folder, run_id)

    os.makedirs(run_folder, exist_ok=True)

    # --------------------------
    # Crear entornos
    # --------------------------
    obs_enum = ObservationType(config["observation_type"])
    act_enum = ActionType(config["action_type"])
    multiagent = bool(config["multiagent"])
    num_drones = int(config["num_drones"]) if multiagent else 1
    seed = int(config["seed"])

    if not multiagent:
        train_env = make_vec_env(
            HoverAviary,
            env_kwargs=dict(obs=obs_enum, act=act_enum),
            n_envs=1,
            seed=seed
        )
        eval_env = HoverAviary(obs=obs_enum, act=act_enum)
    else:
        train_env = make_vec_env(
            MultiHoverAviary,
            env_kwargs=dict(num_drones=num_drones, obs=obs_enum, act=act_enum),
            n_envs=1,
            seed=seed
        )
        eval_env = MultiHoverAviary(num_drones=num_drones, obs=obs_enum, act=act_enum)

    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # --------------------------
    # Modelo PPO
    # --------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(config["learning_rate"]),
        n_steps=int(config["n_steps"]),
        batch_size=int(config["batch_size"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        verbose=1,
        device=config["device"],
        seed=seed
    )

    # --------------------------
    # Target reward (si no viene en config)
    # --------------------------
    target_reward = config.get("target_reward", None)
    if target_reward is None:
        if config["action_type"].lower() == "one_d_rpm":
            target_reward = 474.0 if not multiagent else 949.5
        else:
            target_reward = 467.0 if not multiagent else 920.0

    use_reward_threshold_stop = bool(config["use_reward_threshold_stop"])

    callback_on_best = None
    if use_reward_threshold_stop and target_reward is not None:
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=float(target_reward),
            verbose=1
        )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=run_folder + '/',
        log_path=run_folder + '/',
        eval_freq=int(config["eval_freq"]),
        deterministic=True,
        render=False
    )

    # --------------------------
    # Entrenamiento
    # --------------------------
    total_timesteps = int(config["total_timesteps"])

    print(f"[INFO] Starting training for {total_timesteps} timesteps")
    train_start_wall = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=100
    )
    train_end_wall = time.time()
    wall_time_training_sec = train_end_wall - train_start_wall
    timesteps_total = int(model.num_timesteps)

    # Guardar modelo final
    final_model_path = os.path.join(run_folder, 'final_model.zip')
    model.save(final_model_path)
    print("[INFO] Final model saved to", final_model_path)

    # --------------------------
    # Cargar mejor modelo (si existe)
    # --------------------------
    best_model_path = os.path.join(run_folder, 'best_model.zip')
    if os.path.isfile(best_model_path):
        path_model_to_use = best_model_path
    else:
        print("[WARN] No best_model.zip found, using final_model.zip")
        path_model_to_use = final_model_path

    model = PPO.load(path_model_to_use)
    print("[INFO] Loaded model from", path_model_to_use)

    # --------------------------
    # Leer evaluaciones de entrenamiento
    # --------------------------
    eval_file = os.path.join(run_folder, 'evaluations.npz')
    training_metrics = {}
    if os.path.isfile(eval_file):
        with np.load(eval_file) as data:
            timesteps_arr = data["timesteps"]               # shape (n_eval,)
            results_matrix = data["results"]                # shape (n_eval, n_eval_episodes)
            results_mean = np.mean(results_matrix, axis=1)  # mean reward por evaluación

        if results_mean.size > 0:
            best_idx = int(np.argmax(results_mean))
            best_eval_mean_reward = float(results_mean[best_idx])
            best_eval_std_reward = float(np.std(results_matrix[best_idx]))
            timesteps_at_best = int(timesteps_arr[best_idx])

            final_eval_mean_reward = float(results_mean[-1])
            final_eval_std_reward = float(np.std(results_matrix[-1]))
            auc_normalized = float(np.mean(results_mean))

            timesteps_to_target = None
            if target_reward is not None:
                mask = results_mean >= float(target_reward)
                if np.any(mask):
                    timesteps_to_target = int(timesteps_arr[mask][0])

            stopped_early = timesteps_total < total_timesteps

            training_metrics = {
                "timesteps_total": timesteps_total,
                "wall_time_training_sec": wall_time_training_sec,
                "best_eval_mean_reward": best_eval_mean_reward,
                "best_eval_std_reward": best_eval_std_reward,
                "timesteps_at_best": timesteps_at_best,
                "final_eval_mean_reward": final_eval_mean_reward,
                "final_eval_std_reward": final_eval_std_reward,
                "auc_normalized": auc_normalized,
                "stopped_early": stopped_early,
                "timesteps_to_target": timesteps_to_target
            }

            # (Opcional) imprimir curva si estás en local y quieres verla
            if config.get("plot", False):
                plt.plot(timesteps_arr, results_mean, marker='o', linestyle='-', markersize=4)
                plt.xlabel('Training Steps')
                plt.ylabel('Eval Episode Reward (mean)')
                plt.grid(True, alpha=0.6)
                plt.show()
        else:
            training_metrics = {
                "timesteps_total": timesteps_total,
                "wall_time_training_sec": wall_time_training_sec
            }
    else:
        print("[WARN] evaluations.npz not found, training metrics limited.")
        training_metrics = {
            "timesteps_total": timesteps_total,
            "wall_time_training_sec": wall_time_training_sec
        }

    # --------------------------
    # Evaluación en test
    # --------------------------
    if not multiagent:
        test_env = HoverAviary(
            gui=bool(config["gui"]),
            obs=obs_enum,
            act=act_enum,
            record=bool(config["record_video"])
        )
        test_env_nogui = HoverAviary(obs=obs_enum, act=act_enum)
    else:
        test_env = MultiHoverAviary(
            gui=bool(config["gui"]),
            num_drones=num_drones,
            obs=obs_enum,
            act=act_enum,
            record=bool(config["record_video"])
        )
        test_env_nogui = MultiHoverAviary(
            num_drones=num_drones,
            obs=obs_enum,
            act=act_enum
        )

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=num_drones,
        output_folder=run_folder,
        colab=bool(config["colab"])
    )

    # Reward medio de test (SB3)
    n_test_episodes = int(config["n_test_episodes"])
    test_return_mean, test_return_std = evaluate_policy(
        model,
        test_env_nogui,
        n_eval_episodes=n_test_episodes
    )
    print(f"[INFO] Test mean reward: {test_return_mean} +- {test_return_std}")

    # Métricas de comportamiento (posición + suavidad de acciones)
    # Para no liarla con formatos raros, las métricas de posición solo se calculan en:
    #   - single-agent
    #   - obs = KIN
    track_position = (not multiagent) and (obs_enum == ObservationType.KIN)

    max_steps = int((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ)

    pos_rmse_list = []
    pos_max_list = []
    smoothness_list = []
    crash_count = 0

    pos_target = np.array(config.get("pos_target", [0.0, 0.0, 1.0]), dtype=float)

    for ep in range(n_test_episodes):
        obs, info = test_env.reset(seed=seed + ep, options={})
        episode_pos_errors = []
        episode_action_deltas = []
        prev_action = None
        terminated = False
        truncated = False

        for t in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)

            obs2 = np.array(obs).squeeze()
            act2 = np.array(action).squeeze()

            # Error de posición
            if track_position:
                pos = obs2[0:3]
                pos_err = float(np.linalg.norm(pos - pos_target))
                episode_pos_errors.append(pos_err)

            # Suavidad de acción
            if prev_action is not None:
                delta = float(np.linalg.norm(act2 - prev_action))
                episode_action_deltas.append(delta)
            prev_action = act2

            # Logging para el Logger (solo KIN)
            if obs_enum == ObservationType.KIN:
                if not multiagent:
                    logger.log(
                        drone=0,
                        timestamp=t / test_env.CTRL_FREQ,
                        state=np.hstack([obs2[0:3],
                                         np.zeros(4),
                                         obs2[3:15],
                                         act2]),
                        control=np.zeros(12)
                    )
                else:
                    for d in range(num_drones):
                        logger.log(
                            drone=d,
                            timestamp=t / test_env.CTRL_FREQ,
                            state=np.hstack([obs2[d][0:3],
                                             np.zeros(4),
                                             obs2[d][3:15],
                                             act2[d]]),
                            control=np.zeros(12)
                        )

            if bool(config["gui"]):
                test_env.render()

            # En modo headless, no queremos ralentizar con sync, así que no lo llamamos

            if terminated or truncated:
                break

        # Heurística simple de "crash": terminó antes del límite de pasos
        if terminated and t < max_steps - 1:
            crash_count += 1

        # Agregar métricas episodio
        if episode_pos_errors:
            rmse = float(np.sqrt(np.mean(np.square(episode_pos_errors))))
            max_err = float(np.max(episode_pos_errors))
        else:
            rmse = None
            max_err = None

        if episode_action_deltas:
            smoothness = float(np.mean(episode_action_deltas))
        else:
            smoothness = None

        pos_rmse_list.append(rmse)
        pos_max_list.append(max_err)
        smoothness_list.append(smoothness)

    test_env.close()

    test_pos_error_rmse_mean, test_pos_error_rmse_std = safe_mean_std(pos_rmse_list)
    test_pos_error_max_mean, _ = safe_mean_std(pos_max_list)
    test_action_smoothness_mean, test_action_smoothness_std = safe_mean_std(smoothness_list)
    test_crash_rate = (crash_count / n_test_episodes) if n_test_episodes > 0 else None

    if config.get("plot", False) and obs_enum == ObservationType.KIN:
        logger.plot()

    # --------------------------
    # Preparar JSON de métricas
    # --------------------------
    script_end = datetime.now()

    config_for_json = dict(config)
    config_for_json["target_reward_resolved"] = float(target_reward) if target_reward is not None else None
    config_for_json["run_folder"] = run_folder
    config_for_json["model_file_used_for_test"] = os.path.basename(path_model_to_use)

    test_metrics = {
        "test_return_mean": float(test_return_mean),
        "test_return_std": float(test_return_std),
        "test_pos_error_rmse_mean": test_pos_error_rmse_mean,
        "test_pos_error_rmse_std": test_pos_error_rmse_std,
        "test_pos_error_max_mean": test_pos_error_max_mean,
        "test_action_smoothness_mean": test_action_smoothness_mean,
        "test_action_smoothness_std": test_action_smoothness_std,
        "test_crash_rate": test_crash_rate
    }

    metrics = {
        "run_id": run_id,
        "timestamp_start": script_start.isoformat(),
        "timestamp_end": script_end.isoformat(),
        "config": config_for_json,
        "training_metrics": training_metrics,
        "test_metrics": test_metrics
    }

    # --------------------------
    # Guardar métricas en JSON
    # --------------------------
    metrics_dir = os.path.dirname(metrics_output_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Metrics JSON written to", metrics_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single/multi agent RL experiment for gym-pybullet-drones with JSON config')
    parser.add_argument(
        '--input-dir',
        default=DEFAULT_INPUT_DIR,
        type=str,
        help='Input directory (not used currently, kept for workflow compatibility)'
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        type=str,
        help='Base directory for outputs (default: /data/outputs)'
    )
    parser.add_argument(
        '--config-dir',
        default=DEFAULT_CONFIG_DIR,
        type=str,
        help='Directory containing the config JSON (default: /data/config)'
    )
    parser.add_argument(
        '--config',
        default=None,
        type=str,
        help='Path to input JSON config file (default: /data/config/variables.json)'
    )
    parser.add_argument(
        '--metrics_output',
        default=None,
        type=str,
        help='Path to output JSON metrics file (default: /data/outputs/metrics.json)'
    )
    ARGS = parser.parse_args()

    output_dir = Path(ARGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dir = Path(ARGS.config_dir)
    config_path = resolve_with_base(config_dir, ARGS.config, DEFAULT_CONFIG_FILE)
    metrics_output_path = resolve_with_base(output_dir, ARGS.metrics_output, DEFAULT_METRICS_FILE)

    cfg = load_config(str(config_path))
    cfg["output_folder"] = resolve_output_folder(cfg.get("output_folder"), output_dir)
    run(cfg, metrics_output_path=str(metrics_output_path))
