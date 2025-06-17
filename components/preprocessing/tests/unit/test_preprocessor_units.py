import pytest
import pandas as pd
import numpy as np
import yaml
import tempfile
import os
from pathlib import Path

# Assuming preprocess.py is in components/preprocessing/
# and this test file is in components/preprocessing/tests/unit/
# Adjust import if preprocess.py is moved to a 'microservice' subfolder later
from components.preprocessing.preprocess import TimeSeriesPreprocessor, load_config

@pytest.fixture
def base_config():
    """Provides a base configuration for tests."""
    return {
        'logging': {'level': 'WARNING'},
        'data': {'datetime_column': None}, # Keep None for most unit tests, can be overridden
        'preprocessing': {
            'features': {},
            'missing_values': {
                'strategy': 'fill',
                'method': 'interpolate'
            },
            'outliers': {
                'enabled': False, # Default to False for unit tests unless testing outliers
                'method': 'iqr',
                'action': 'cap'
            },
            'normalization': {
                'enabled': False, # Default to False for unit tests unless testing normalization
                'method': 'minmax'
            },
            'resampling': {
                'enabled': False
            },
            'sliding_windows': {
                'enabled': False # Default to False for unit tests unless testing sliding_windows
            }
        }
    }

@pytest.fixture
def preprocessor(base_config):
    """Provides a TimeSeriesPreprocessor instance with a base config."""
    return TimeSeriesPreprocessor(base_config)

def test_load_config_valid_yaml(tmp_path):
    """Test loading a valid YAML configuration file."""
    test_config_data = {
        'preprocessing': {
            'missing_values': {'strategy': 'fill'},
            'outliers': {'enabled': False}
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config_data, f)

    loaded_config = load_config(str(config_file))
    assert loaded_config['preprocessing']['missing_values']['strategy'] == 'fill'
    assert not loaded_config['preprocessing']['outliers']['enabled']

def test_load_config_invalid_path():
    """Test loading a non-existent configuration file."""
    with pytest.raises(SystemExit): # As per original load_config error handling
        load_config("non_existent_config.yaml")

# Unit test for feature selection
def test_select_features_include_exclude(preprocessor, base_config):
    """Test feature selection with include and exclude lists."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'feature1': [1, 2],
        'feature2': [3, 4],
        'feature_to_exclude': [5, 6],
        'another_feature': [7, 8]
    }
    df_input = pd.DataFrame(data)

    # Update config for this test
    current_config = base_config.copy()
    current_config['preprocessing']['features'] = {
        'include': ['timestamp', 'feature1', 'another_feature'],
        'exclude': ['another_feature'] # feature_to_exclude is not in include, so it's already out
    }

    # Create a new preprocessor with the updated config for this specific test
    custom_preprocessor = TimeSeriesPreprocessor(current_config)

    df_output = custom_preprocessor.select_features(df_input.copy()) # Use copy to avoid modifying original

    assert 'feature1' in df_output.columns
    assert 'timestamp' in df_output.columns
    assert 'feature_to_exclude' not in df_output.columns
    assert 'another_feature' not in df_output.columns # Excluded
    assert 'feature2' not in df_output.columns # Not in include
    assert len(df_output.columns) == 2


def test_select_features_only_include(preprocessor, base_config):
    data = pd.DataFrame({
        'A': [1, 2], 'B': [3, 4], 'C': [5, 6]
    })
    config = base_config.copy()
    config['preprocessing']['features'] = {'include': ['A', 'C']}
    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_result = custom_preprocessor.select_features(data.copy())
    assert list(df_result.columns) == ['A', 'C']

def test_select_features_only_exclude(preprocessor, base_config):
    data = pd.DataFrame({
        'A': [1, 2], 'B': [3, 4], 'C': [5, 6]
    })
    config = base_config.copy()
    config['preprocessing']['features'] = {'exclude': ['B']}
    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_result = custom_preprocessor.select_features(data.copy())
    assert list(df_result.columns) == ['A', 'C']

def test_select_features_no_config(preprocessor): # Uses default preprocessor
    data = pd.DataFrame({
        'A': [1, 2], 'B': [3, 4]
    })
    df_result = preprocessor.select_features(data.copy()) # Should do nothing
    assert list(df_result.columns) == ['A', 'B']

# Unit tests for missing values
def test_handle_missing_values_fill_interpolate(preprocessor, base_config):
    """Test missing value handling with interpolate strategy."""
    data = {'value': [10, np.nan, 20, np.nan, 30]}
    df_input = pd.DataFrame(data)

    config = base_config.copy()
    config['preprocessing']['missing_values']['strategy'] = 'fill'
    config['preprocessing']['missing_values']['method'] = 'interpolate'

    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_output = custom_preprocessor.handle_missing_values(df_input.copy())

    assert df_output['value'].isnull().sum() == 0
    assert df_output.loc[1, 'value'] == 15 # Interpolated between 10 and 20
    assert df_output.loc[3, 'value'] == 25 # Interpolated between 20 and 30

def test_handle_missing_values_fill_mean(preprocessor, base_config):
    """Test missing value handling with fill mean strategy."""
    data = {'value': [10, 20, np.nan, 40]}
    df_input = pd.DataFrame(data)

    config = base_config.copy()
    config['preprocessing']['missing_values']['strategy'] = 'fill'
    config['preprocessing']['missing_values']['method'] = 'mean'

    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_output = custom_preprocessor.handle_missing_values(df_input.copy())

    expected_mean = (10 + 20 + 40) / 3.0
    assert df_output['value'].isnull().sum() == 0
    assert round(df_output.loc[2, 'value'], 5) == round(expected_mean, 5)

def test_handle_missing_values_drop_strategy(preprocessor, base_config):
    """Test missing value handling with drop strategy."""
    data = {
        'col1': [1, np.nan, 3, 4, 5],
        'col2': [np.nan, 'b', 'c', 'd', 'e'], # Non-numeric to test drop
        'col3': [10, 20, 30, np.nan, 50]
    }
    df_input = pd.DataFrame(data)

    config = base_config.copy()
    config['preprocessing']['missing_values']['strategy'] = 'drop'
    # Drop row if it does not have at least 2 non-NA values (3 cols total)
    config['preprocessing']['missing_values']['threshold'] = 2/3

    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_output = custom_preprocessor.handle_missing_values(df_input.copy())

    # Row 0: col1=1, col2=nan, col3=10 (2 non-NA) -> keep
    # Row 1: col1=nan, col2='b', col3=20 (2 non-NA) -> keep
    # Row 2: col1=3, col2='c', col3=30 (3 non-NA) -> keep
    # Row 3: col1=4, col2='d', col3=nan (2 non-NA) -> keep
    # Row 4: col1=5, col2='e', col3=50 (3 non-NA) -> keep
    # The threshold in preprocess.py is `df.dropna(thresh=int(len(df.columns) * threshold))`
    # For 3 columns, threshold 2/3 means int(3 * 2/3) = int(2) = 2. So rows with at least 2 non-NA are kept.
    # The original test_missing_values_strategies used threshold 1.0 for a single column df, which meant drop all rows with NaN.
    # Let's make a more specific test for drop:
    data_single_col = {'value': [1, np.nan, 3, np.nan, 5]}
    df_single_col_input = pd.DataFrame(data_single_col)

    config_single_col = base_config.copy()
    config_single_col['preprocessing']['missing_values']['strategy'] = 'drop'
    # For a single column, if threshold is 1.0, it means row must have at least 1 non-NA value.
    # If threshold is 0.5 (meaning 0.5 * 1 col = 0 non-NA values required), it keeps everything.
    # The original test had: config['preprocessing']['missing_values']['threshold'] = 1.0
    # This means for a single column df, it drops rows with NA.
    config_single_col['preprocessing']['missing_values']['threshold'] = 1.0

    custom_preprocessor_single_col = TimeSeriesPreprocessor(config_single_col)
    df_single_col_output = custom_preprocessor_single_col.handle_missing_values(df_single_col_input.copy())

    assert len(df_single_col_output) == 3
    assert df_single_col_output['value'].isnull().sum() == 0


# Unit tests for outlier detection (basic functionality, not full process)
def test_detect_outliers_iqr_cap(preprocessor, base_config):
    """Test outlier detection with IQR method and cap action."""
    data = {'value': [10, 12, 11, 9, 100, 8, 13]} # 100 is an outlier
    df_input = pd.DataFrame(data)

    config = base_config.copy()
    config['preprocessing']['outliers']['enabled'] = True
    config['preprocessing']['outliers']['method'] = 'iqr'
    config['preprocessing']['outliers']['action'] = 'cap'

    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_output = custom_preprocessor.detect_outliers(df_input.copy())

    # Calculate Q1, Q3, IQR for [10, 12, 11, 9, 8, 13] (excluding 100 for bounds calculation)
    # Or rather, the function calculates it on the whole series including the outlier
    # Q1 = 9.5, Q3 = 12.5, IQR = 3
    # Lower bound = 9.5 - 1.5 * 3 = 5
    # Upper bound = 12.5 + 1.5 * 3 = 17
    # So 100 should be capped to 17
    # The original value 100 was at index 4
    assert df_output.loc[4, 'value'] == 17.0
    assert df_output['value'].max() <= 17.0

# Unit tests for normalization
def test_normalize_data_minmax(preprocessor, base_config):
    """Test data normalization with MinMax method."""
    data = {'value': [10, 20, 30, 40, 50]}
    df_input = pd.DataFrame(data)

    config = base_config.copy()
    config['preprocessing']['normalization']['enabled'] = True
    config['preprocessing']['normalization']['method'] = 'minmax'

    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_output = custom_preprocessor.normalize_data(df_input.copy())

    assert df_output['value'].min() == 0.0
    assert df_output['value'].max() == 1.0
    assert df_output.loc[2, 'value'] == 0.5 # (30-10)/(50-10) = 20/40 = 0.5

# Unit test for datetime processing
def test_preprocess_datetime_auto_detect(preprocessor, base_config):
    data = {'my_timestamp': ['2023-01-01 10:00', '2023-01-01 12:00'], 'value': [1,2]}
    df_input = pd.DataFrame(data)

    # Config does not specify datetime_column, so it should be auto-detected
    custom_preprocessor = TimeSeriesPreprocessor(base_config.copy())
    df_output = custom_preprocessor.preprocess_datetime(df_input.copy())

    assert isinstance(df_output.index, pd.DatetimeIndex)
    assert df_output.index.name == 'my_timestamp'

def test_preprocess_datetime_specified_column(preprocessor, base_config):
    data = {'event_date': ['2023-01-01', '2023-01-02'], 'value': [1,2]}
    df_input = pd.DataFrame(data)

    config = base_config.copy()
    config['data']['datetime_column'] = 'event_date'
    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_output = custom_preprocessor.preprocess_datetime(df_input.copy())

    assert isinstance(df_output.index, pd.DatetimeIndex)
    assert df_output.index.name == 'event_date'

def test_resample_data_sum(preprocessor, base_config):
    idx = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:30', '2023-01-01 11:00'])
    data = {'value': [10, 5, 20]}
    df_input = pd.DataFrame(data, index=idx)

    config = base_config.copy()
    config['preprocessing']['resampling']['enabled'] = True
    config['preprocessing']['resampling']['frequency'] = 'h' # Resample to hourly
    config['preprocessing']['resampling']['method'] = 'sum'

    custom_preprocessor = TimeSeriesPreprocessor(config)
    # Ensure preprocess_datetime has been called or index is already datetime
    # For unit test, we can directly pass df with DatetimeIndex
    df_output = custom_preprocessor.resample_data(df_input.copy())

    assert len(df_output) == 2 # 10:00 and 11:00 hours
    assert df_output.loc[pd.to_datetime('2023-01-01 10:00'), 'value'] == 15 # 10 + 5
    assert df_output.loc[pd.to_datetime('2023-01-01 11:00'), 'value'] == 20

def test_create_sliding_windows(preprocessor, base_config):
    data = {'value': [1,2,3,4,5]}
    df_input = pd.DataFrame(data)

    config = base_config.copy()
    config['preprocessing']['sliding_windows']['enabled'] = True
    config['preprocessing']['sliding_windows']['size'] = 3

    custom_preprocessor = TimeSeriesPreprocessor(config)
    df_output = custom_preprocessor.create_sliding_windows(df_input.copy())

    assert 'value_rolling_mean' in df_output.columns
    assert 'value_rolling_std' in df_output.columns
    # First two values of rolling operations will be NaN for window size 3
    assert pd.isna(df_output.loc[0, 'value_rolling_mean'])
    assert pd.isna(df_output.loc[1, 'value_rolling_mean'])
    assert df_output.loc[2, 'value_rolling_mean'] == 2.0 # Mean of (1,2,3)
    assert df_output.loc[4, 'value_rolling_mean'] == 4.0 # Mean of (3,4,5)
