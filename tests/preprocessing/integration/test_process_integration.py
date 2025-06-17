import pandas as pd
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3] / 'components' / 'preprocessing'))
from preprocess import TimeSeriesPreprocessor


def test_process_end_to_end(tmp_path):
    config = {
        'logging': {'level': 'WARNING'},
        'preprocessing': {
            'resampling': {'enabled': True, 'frequency': '1D', 'method': 'sum'}
        }
    }
    config_path = tmp_path / 'config.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    df = pd.DataFrame({'timestamp': ['2024-01-01 10:00', '2024-01-01 12:00', '2024-01-02 08:00'],
                       'sales': [10, 5, 20]})
    input_path = tmp_path / 'data.csv'
    df.to_csv(input_path, index=False)
    output_path = tmp_path / 'result.csv'

    pre = TimeSeriesPreprocessor(config)
    pre.process(str(input_path), str(output_path))

    assert output_path.exists()
    result = pd.read_csv(output_path)
    assert result.shape[0] == 2
    assert result.loc[0, 'sales'] == 15
