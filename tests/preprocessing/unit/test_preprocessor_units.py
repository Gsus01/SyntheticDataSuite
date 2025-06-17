import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add preprocess module to path
sys.path.append(str(Path(__file__).resolve().parents[3] / 'components' / 'preprocessing'))
from preprocess import TimeSeriesPreprocessor


def test_handle_missing_fill_mean():
    config = {
        'logging': {'level': 'WARNING'},
        'preprocessing': {
            'missing_values': {
                'strategy': 'fill',
                'method': 'mean'
            }
        }
    }
    df = pd.DataFrame({'value': [10, 20, np.nan, 40]})
    pre = TimeSeriesPreprocessor(config)
    result = pre.handle_missing_values(df)
    assert result['value'].isnull().sum() == 0
    assert round(result.loc[2, 'value'], 1) == round((10+20+40)/3, 1)
