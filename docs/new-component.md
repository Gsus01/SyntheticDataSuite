# 🐳 Container Integration Guide for Synthetic Data Suite

This guide outlines the standard protocol for developing containerized components (Data Generators, Preprocessors, Trainers) compatible with the Synthetic Data Suite workflow engine.

---

## ⚡ The "Golden Rule": Directory Protocol

To ensure seamless integration with our Argo Workflows orchestrator, all containers must adhere to the **Standard Directory Protocol**.

**Your application should generally scan the standard input directory or expect specific filenames within it.**

| Directory   | Path Inside Container | Description                                                  |
| :---------- | :-------------------- | :----------------------------------------------------------- |
| **Inputs**  | `/data/inputs`        | The workflow will mount input artifacts (CSVs, Models) here. |
| **Outputs** | `/data/outputs`       | Write all your results (datasets, logs, models) here.        |
| **Config**  | `/data/config`        | (Optional) JSON/YAML configuration files will be mounted here. |

---

## 🐍 Python Implementation Guide

Your scripts must support **Dual Mode**:

1. **Local Development:** Allows passing custom paths via arguments.
2. **Production (Cluster):** Uses standard default paths without arguments.

### ✅ Basic Pattern (Standard Inputs)

Use `argparse` to define arguments but set the **default values** to the standard directories.

```python
import argparse
import pandas as pd
from pathlib import Path

# 1. Define Standard Paths as Defaults
DEFAULT_INPUT_DIR = "/data/inputs"
DEFAULT_OUTPUT_DIR = "/data/outputs"

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Suite Component")
    
    # 2. Use arguments with defaults. 
    # This allows: 'python script.py' (Production) AND 'python script.py --input-dir ./local' (Dev)
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Your Logic Here
    # ...

if __name__ == "__main__":
    main()
```

---

## 📂 Advanced: Handling Multiple Inputs

When your component needs to read input files, choose the strategy that matches your use case.

### Strategy A: Specific Files (The "Contract" Approach)

**Use when:** You need distinct files (e.g., `train.csv` AND `test.csv`).

**Rule:** Hardcode the *expected filenames* in your script. The workflow orchestrator (Argo) is responsible for renaming incoming artifacts to match these names.

```python
# Inside main():

# CONTRACT: We explicitly expect these two files to exist in the input folder.
train_path = input_dir / "train_data.csv"
test_path = input_dir / "test_data.csv"

if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        f"Missing required files in {input_dir}. "
        "Expected 'train_data.csv' and 'test_data.csv'."
    )

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
```

### Strategy B: Batch Processing (The "Discovery" Approach)

**Use when:** You want to process *all* files in the folder, regardless of name (e.g., processing 50 chunks of data).

**Rule:** Use `glob` to find files dynamically.

```python
# Inside main():

# DISCOVERY: Find all CSVs
files = list(input_dir.glob("*.csv"))

if not files:
    print("No CSV files found.")
    return

for file_path in files:
    print(f"Processing {file_path.name}...")
    process_file(file_path)
```

---

## 🐳 Dockerfile Standards

Your Dockerfile should be "clean" and rely on the Python script's default arguments.

### ❌ DON'T (Hardcoded Entrypoints)

```dockerfile
# Avoid this! It makes local testing difficult.
ENTRYPOINT ["python", "main.py", "--input-dir", "/data/inputs"] 
```

### ✅ DO (Clean Command)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Just run the script. It will use the default /data/inputs paths automatically.
CMD ["python", "main.py"]
```

---

## 🧪 How to Test Locally

Since your script supports arguments, you don't need to create `/data/inputs` on your laptop. Just override the paths:

```bash
# Testing on your machine with custom folders
python my_component/main.py \
    --input-dir ./test_data/in \
    --output-dir ./test_data/out
```

When deployed to the cluster, the system will simply run `python main.py`, and the script will correctly look into `/data/inputs`.

---

## 🔧 Registry Integration (DB-backed)

The component catalog is now dynamic: components are registered at runtime via API and stored in PostgreSQL.

Key ideas:
- You register one **ComponentSpec** (a JSON object) per component version.
- The backend stores the spec in Postgres (tables `components` and `component_versions`).
- The UI loads the catalog from the backend (GET `/workflow-templates`). There are no YAML files to edit.
- By default the system uses the **active** version (active/latest) for each component.

### 1) What you need to prepare

- A Docker image for your component (image names must be lowercase; Docker rejects uppercase).
- A ComponentSpec (you can author it in YAML for convenience, but you register it by sending JSON to the API).

### 2) ComponentSpec fields and meaning

Required fields (minimum viable):
- `apiVersion`: `sds/v1`
- `kind`: `Component`
- `metadata.name`: component identifier (unique; kebab-case recommended)
- `metadata.version`: version (e.g. `v1`, `1.0.0`, etc.)
- `metadata.type`: `input | preprocessing | training | generation | output | other`
- `io.inputs[]` and `io.outputs[]`: ports/artifacts with `name`, `path` and optional `role`
- `runtime.image`: Docker image to run
- `runtime.command`/`runtime.args`: how to run it (if your Dockerfile already defines `CMD`, you can omit `command/args`)

About `io.*.role`:
- `role: config` means this input is a config file (not connectable in the canvas; the system generates it from `parameters.defaults`).
- `role: data` (default) means a connectable artifact.

About `parameters.defaults`:
- A JSON object with default values shown in the inspector.
- At this stage we are not using JSON Schema; it is defaults only.

### 3) Minimal example (1 input, 1 config, 1 output)

```json
{
  "apiVersion": "sds/v1",
  "kind": "Component",
  "metadata": {
    "name": "train-my-model",
    "version": "v1",
    "type": "training",
    "title": "Train My Model"
  },
  "io": {
    "inputs": [
      {"name": "processed-data", "path": "/data/inputs/preprocessed_input.csv", "role": "data"},
      {"name": "training-config", "path": "/data/config/training.json", "role": "config"}
    ],
    "outputs": [
      {"name": "trained-model", "path": "/data/outputs/model.pkl", "role": "model"}
    ]
  },
  "runtime": {
    "image": "docker.io/library/training-my_model:latest",
    "imagePullPolicy": "Never",
    "command": ["python", "main.py"]
  },
  "parameters": {
    "defaults": {
      "epochs": 10,
      "seed": 123
    }
  }
}
```

### 4) Register the component (API)

Endpoint payload:
- `spec`: the full ComponentSpec
- `activate`: if `true`, this version becomes the active (latest) version

```bash
curl -X POST http://localhost:8000/components \
  -H 'Content-Type: application/json' \
  -d '{
    "spec": {
      "apiVersion": "sds/v1",
      "kind": "Component",
      "metadata": {"name": "train-my-model", "version": "v1", "type": "training", "title": "Train My Model"},
      "io": {
        "inputs": [
          {"name": "processed-data", "path": "/data/inputs/preprocessed_input.csv", "role": "data"},
          {"name": "training-config", "path": "/data/config/training.json", "role": "config"}
        ],
        "outputs": [
          {"name": "trained-model", "path": "/data/outputs/model.pkl", "role": "model"}
        ]
      },
      "runtime": {"image": "docker.io/library/training-my_model:latest", "imagePullPolicy": "Never", "command": ["python", "main.py"]},
      "parameters": {"defaults": {"epochs": 10}}
    },
    "activate": true
  }'
```

### 5) Activate another version (active/latest)

```bash
curl -X POST http://localhost:8000/components/train-my-model/v2/activate
```

### 6) Verify it is registered

```bash
curl http://localhost:8000/components
curl http://localhost:8000/workflow-templates
```

- `GET /components` lists components and their `activeVersion`.
- `GET /workflow-templates` is what the UI consumes to display the catalog.

### Naming Conventions

| Element | Convention | Example |
| :------ | :--------- | :------ |
| Component name | kebab-case | `train-hmm-model`, `generate-copulas-data` |
| Image name | lowercase, no spaces | `training-hmm:latest`, `generation-copulas:latest` |
| Config artifact | `{something}-config` | `training-hmm-config` |
| Directory | `components/{type}/{model}/` | `components/training/hmm/` |

---

## 📝 Integration Checklist

**Container Requirements**

- [ ] Script accepts `--input-dir` and `--output-dir` with defaults `/data/inputs` and `/data/outputs`
- [ ] Writes all outputs to `/data/outputs`
- [ ] If using config, reads from `/data/config`
- [ ] Dockerfile does not hardcode runtime paths (use `CMD ["python", "main.py"]` or equivalent)

**Registry Requirements**

- [ ] `metadata.name` is unique and stable
- [ ] `metadata.version` increments when the contract changes
- [ ] `io.inputs[].path` and `io.outputs[].path` match what your container actually reads/writes
- [ ] `runtime.image` exists in minikube/cluster (and is lowercase)
- [ ] Registered via `POST /components` and visible in `GET /workflow-templates`

**Smoke test**

- [ ] Appears in the canvas
- [ ] You can upload a file using `data-input`
- [ ] The workflow compiles and runs in Argo

