import subprocess
import pandas as pd
import yaml
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = ROOT / 'components' / 'preprocessing' / 'docker-compose.yml'


def test_container_as_black_box(tmp_path):
    config = {
        'logging': {'level': 'WARNING'},
        'preprocessing': {
            'missing_values': {'strategy': 'fill', 'method': 'mean'}
        }
    }
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    df = pd.DataFrame({'timestamp': ['2024-01-01', '2024-01-02'],
                       'value': [1, None]})
    data_path = tmp_path / 'data.csv'
    df.to_csv(data_path, index=False)

    output_dir = tmp_path / 'out'
    output_dir.mkdir()
    output_file = output_dir / 'result.csv'

    cmd = [
        'docker', 'compose', '-f', str(COMPOSE_FILE), 'run', '--rm',
        '-v', f'{config_path}:/app/config.yaml:ro',
        '-v', f'{data_path}:/app/data/input/data.csv:ro',
        '-v', f'{output_dir}:/app/data/output',
        'timeseries-preprocessor',
        '--config', '/app/config.yaml',
        '--input', '/app/data/input/data.csv',
        '--output', '/app/data/output/result.csv'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output_file.exists()
    processed = pd.read_csv(output_file)
    assert processed['value'].isnull().sum() == 0
