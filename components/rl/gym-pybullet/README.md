# gym-pybullet-drones RL component

Headless training/evaluation wrapper for `gym_pybullet_drones` using Stable-Baselines3 PPO. Default paths follow the platform convention (`/data/config`, `/data/outputs`).

## Build (local/Minikube)

```bash
docker build -t rl-gym-pybullet:latest components/rl/gym-pybullet
```

## Run locally with the standard mounts

```bash
mkdir -p /tmp/drones-out
docker run --rm \
  -v $(pwd)/components/rl/gym-pybullet/variables.json:/data/config/variables.json:ro \
  -v /tmp/drones-out:/data/outputs \
  rl-gym-pybullet:latest
```

Artifacts produced:
- `/data/outputs/metrics.json`: aggregated training/test metrics.
- `/data/outputs/results/`: run folder with `best_model.zip`, logs, and SB3 artifacts.

Notes
- The Dockerfile clones `gym-pybullet-drones` (branch `main`) and overlays the local `rl_example.py` and `variables.json` so tweaks to this component are picked up without copying the full upstream repo into the build context.
