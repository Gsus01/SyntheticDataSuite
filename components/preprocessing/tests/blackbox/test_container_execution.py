import pytest
import subprocess
import os
import pandas as pd
from pathlib import Path
import yaml

# Define paths relative to the root of the component (components/preprocessing)
# as docker-compose commands are typically run from there.
COMPONENT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DATA_HOST_DIR = COMPONENT_ROOT / "test_data" # e.g., components/preprocessing/test_data
CONFIG_HOST_PATH = COMPONENT_ROOT / "config.yaml" # e.g., components/preprocessing/config.yaml
DOCKER_COMPOSE_PATH = COMPONENT_ROOT / "docker-compose.yml"

# Define paths as they will appear INSIDE the container, based on typical volume mounts
# These should align with what preprocess.py expects and how docker-compose.yml sets up volumes.
# The example issue used /app/data for inputs/outputs and /app/config.yaml for config.
# We will ensure Step 5 (Update docker-compose.yml) makes these paths valid.
CONTAINER_DATA_DIR = Path("/app/data") # General data directory inside container
CONTAINER_INPUT_DIR = CONTAINER_DATA_DIR / "input"
CONTAINER_OUTPUT_DIR = CONTAINER_DATA_DIR / "output"
CONTAINER_CONFIG_FILE = Path("/app/config.yaml")


@pytest.fixture(scope="module")
def build_docker_image():
    """Ensure the Docker image is built before tests run."""
    # The docker-compose service used for testing should have a `build` context.
    # `docker compose run` will build automatically if image is not found or outdated.
    # For explicit build, one could run:
    # build_command = ["docker", "compose", "-f", str(DOCKER_COMPOSE_PATH), "build", "test-runner"] # Assuming 'test-runner' is the service
    # subprocess.run(build_command, check=True, cwd=COMPONENT_ROOT)
    # However, often `docker compose run` handles this. We'll rely on that for now.
    # It's also assumed the image name in docker-compose.yml is fixed e.g. 'timeseries-preprocessor-testable'
    pass

@pytest.fixture
def host_output_dir(tmp_path_factory):
    """Create a temporary directory on the host for outputs from the container."""
    # Using tmp_path_factory to ensure it's unique per test if needed, or shared if scope is changed.
    return tmp_path_factory.mktemp("blackbox_outputs")


def test_container_process_sensor_data(build_docker_image, host_output_dir):
    """
    Test the container processing sensor_data_with_nulls.csv with default config.
    Assumes a service named 'blackbox-test-runner' is defined in docker-compose.yml
    which mounts necessary volumes.
    """
    input_csv_filename = "sensor_data_with_nulls.csv"
    output_csv_filename = "processed_sensor_data_from_container.csv"

    # Path to the input CSV on the host (will be mapped by docker-compose)
    # For this test, we assume docker-compose.yml maps TEST_DATA_HOST_DIR to CONTAINER_INPUT_DIR

    # Path to the output CSV as it will appear on the HOST after container writes to mapped volume
    host_output_file = host_output_dir / output_csv_filename

    # Clean up previous run output if any
    if host_output_file.exists():
        os.remove(host_output_file)

    # Command to run the preprocessor service via docker-compose
    # Paths for --config, --input, --output are INSIDE the container
    # The service 'blackbox-test-runner' needs to be defined in docker-compose.yml
    # and configured to mount:
    #   - CONFIG_HOST_PATH to CONTAINER_CONFIG_FILE
    #   - TEST_DATA_HOST_DIR to CONTAINER_INPUT_DIR
    #   - host_output_dir to CONTAINER_OUTPUT_DIR
    command = [
        "docker-compose", "-f", str(DOCKER_COMPOSE_PATH), "run", "--rm",
        # "--service-ports", # Remove if not needed, typically for long-running services
        "blackbox-test-runner", # This service needs to be defined in docker-compose.yml
        "-v", f"{str(host_output_dir)}:{str(CONTAINER_OUTPUT_DIR)}",
        "--config", str(CONTAINER_CONFIG_FILE),
        "--input", str(CONTAINER_INPUT_DIR / input_csv_filename),
        "--output", str(CONTAINER_OUTPUT_DIR / output_csv_filename)
    ]

    # Execute the command from the COMPONENT_ROOT where docker-compose.yml is expected
    result = subprocess.run(command, capture_output=True, text=True, cwd=COMPONENT_ROOT)

    print(f"COMMAND: {' '.join(command)}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    assert result.returncode == 0, f"Container execution failed with stderr: {result.stderr}"
    assert host_output_file.exists(), f"Output file {host_output_file} was not created."

    # Basic check on the output file
    result_df = pd.read_csv(host_output_file)
    assert not result_df.empty, "Output CSV is empty."
    assert len(result_df.columns) > 1 # Expect multiple columns after processing

    # (Optional) Clean up the created output file - tmp_path_factory handles the dir
    # if host_output_file.exists():
    #     os.remove(host_output_file)

# Example of a test with a custom configuration
def test_container_custom_config_resample(build_docker_image, host_output_dir, tmp_path_factory):
    input_csv_filename = "irregular_timestamp_data.csv"
    output_csv_filename = "processed_irregular_custom_from_container.csv"
    custom_config_filename = "custom_blackbox_config.yaml"

    # Create a custom config file on the host
    custom_config_host_dir = tmp_path_factory.mktemp("blackbox_custom_config")
    custom_config_host_path = custom_config_host_dir / custom_config_filename

    custom_config_data = {
        'logging': {'level': 'INFO'},
        'data': {'datetime_column': 'timestamp'}, # Explicitly set
        'preprocessing': {
            'resampling': {'enabled': True, 'frequency': 'D', 'method': 'sum'}, # Daily sum
            'missing_values': {'strategy': 'drop'}, # Different from default
            'outliers': {'enabled': False},
            'normalization': {'enabled': False},
            'sliding_windows': {'enabled': False}
        }
    }
    with open(custom_config_host_path, 'w') as f:
        yaml.dump(custom_config_data, f)

    host_output_file = host_output_dir / output_csv_filename
    if host_output_file.exists():
        os.remove(host_output_file)

    # The CONTAINER_CONFIG_FILE path for the custom config needs to be unique if default config is also used by service
    # Or the service needs to be flexible. For this, let's assume custom_config_host_dir is mounted as /app/custom_config_vol
    container_custom_config_path = Path("/app/custom_config_vol") / custom_config_filename
    container_custom_config_vol_str = str(Path("/app/custom_config_vol"))

    command = [
        "docker-compose", "-f", str(DOCKER_COMPOSE_PATH), "run", "--rm",
        "blackbox-test-runner", # This service needs to define a mount for custom_config_host_dir
        "-v", f"{str(host_output_dir)}:{str(CONTAINER_OUTPUT_DIR)}",
        "-v", f"{str(custom_config_host_path.parent)}:{container_custom_config_vol_str}",
        "--config", str(container_custom_config_path),
        "--input", str(CONTAINER_INPUT_DIR / input_csv_filename),
        "--output", str(CONTAINER_OUTPUT_DIR / output_csv_filename)
    ]

    result = subprocess.run(command, capture_output=True, text=True, cwd=COMPONENT_ROOT)

    print(f"COMMAND: {' '.join(command)}")
    print(f"STDOUT: {result.stdout}")
    print(f"STDERR: {result.stderr}")

    assert result.returncode == 0, f"Container execution with custom config failed: {result.stderr}"
    assert host_output_file.exists(), f"Output file {host_output_file} with custom config was not created."

    result_df = pd.read_csv(host_output_file)
    assert not result_df.empty
    # Check if resampling to daily was applied - expect fewer rows than original, unique dates
    original_df = pd.read_csv(TEST_DATA_HOST_DIR / input_csv_filename)
    assert len(result_df) < len(original_df)
    assert pd.to_datetime(result_df['timestamp']).dt.normalize().nunique() == len(result_df)
