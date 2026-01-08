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
- `/data/outputs/best_model.zip`: best-performing checkpoint (falls back to final model if no best is found).
- `/data/outputs/final_model.zip`: last checkpoint after training.
- `/data/outputs/evaluations.npz`: evaluation traces recorded during training.

Notes
- The Dockerfile clones `gym-pybullet-drones` (branch `main`) and overlays the local `rl_example.py` and `variables.json` so tweaks to this component are picked up without copying the full upstream repo into the build context.
- `output_folder` overrides in config are ignored so everything lands next to `metrics.json`.
