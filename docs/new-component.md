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

## 🔧 Repository Integration

Once your container is ready, you need to register it in the repository so that the workflow engine and frontend can use it.

### Files to Modify

| File | Purpose |
| :--- | :------ |
| `backend/catalog/nodes.yaml` | Registers the node in the UI palette |
| `backend/workflow-templates.yaml` | Defines the Argo WorkflowTemplate for execution |
| `components/{type}/{name}/variables.json` | Parameters schema for the frontend form |
| `build_all.sh` | (Optional) Add Docker build command |

### Step 1: Add to Node Catalog

Edit `backend/catalog/nodes.yaml` to add your new node. This makes it appear in the frontend canvas.

```yaml
# backend/catalog/nodes.yaml
nodes:
  # ... existing nodes ...

  - name: train-my-model          # Unique identifier (kebab-case)
    type: training                # One of: input, preprocessing, training, generation, output
    version: v2
    parameters: [params]          # Parameter groups
    parameters_file: components/training/my_model/variables.json
    artifacts:
      inputs:
        - name: processed-data
          path: /data/inputs/preprocessed_input.csv
        - name: training-my-model-config
          path: /data/config/training_my_model.json
      outputs:
        - name: trained-model
          path: /data/outputs/my_model.pkl
    limits: {}
```

### Step 2: Add Workflow Template

Edit `backend/workflow-templates.yaml` to define how Argo runs your container.

```yaml
# backend/workflow-templates.yaml
templates:
  # ... existing templates ...

  train-my-model:
    inputs:
      artifacts:
        - name: processed-data
          path: /data/inputs/preprocessed_input.csv
          archive:
            none: {}
        - name: training-my-model-config
          path: /data/config/training_my_model.json
    container:
      image: docker.io/library/training-my_model:latest
      imagePullPolicy: Never      # Use 'Always' for remote registries
      command: [python, main.py]
    outputs:
      artifacts:
        - name: trained-model
          path: /data/outputs/my_model.pkl
          archive:
            none: {}
```

### Step 3: Create Parameters Schema

Create `components/{type}/{name}/variables.json` to define the configurable parameters.

```json
{
  "n_samples": {
    "type": "integer",
    "default": 1000,
    "description": "Number of samples to generate"
  },
  "learning_rate": {
    "type": "number",
    "default": 0.01,
    "description": "Model learning rate"
  },
  "model_type": {
    "type": "string",
    "default": "standard",
    "enum": ["standard", "advanced"],
    "description": "Type of model to train"
  }
}
```

### Step 4: Build the Docker Image

Add your image to `build_all.sh` or build manually:

```bash
# Manual build
docker build -t training-my_model:latest ./components/training/my_model/

# Or add to build_all.sh
docker build -t training-my_model:latest ./components/training/my_model/
```

### Naming Conventions

| Element | Convention | Example |
| :------ | :--------- | :------ |
| Node name | `{action}-{model}-{type}` | `train-hmm-model`, `generate-copulas-data` |
| Image name | `{type}-{model}:latest` | `training-hmm:latest`, `generation-copulas:latest` |
| Config artifact | `{action}-{model}-config` | `training-hmm-config` |
| Directory | `components/{type}/{model}/` | `components/training/hmm/` |

---

## 📝 Integration Checklist

Before submitting a new component, verify:

**Container Requirements:**

- [ ] Script accepts `--input-dir` and `--output-dir` arguments
- [ ] Defaults are set to `/data/inputs` and `/data/outputs`
- [ ] Uses **Strategy A** (fixed names) or **Strategy B** (glob) for inputs
- [ ] Dockerfile `CMD` runs the script without hardcoded flags
- [ ] `requirements.txt` is included and minimal

**Repository Integration:**

- [ ] Node added to `backend/catalog/nodes.yaml`
- [ ] Template added to `backend/workflow-templates.yaml`
- [ ] `variables.json` created with parameter schema
- [ ] Docker image builds successfully
- [ ] Artifact paths match between catalog and template

**Testing:**

- [ ] Container runs locally with test data
- [ ] Node appears in frontend canvas
- [ ] Workflow executes successfully in Argo
