import pytest
import pandas as pd
import numpy as np

def test_environment_setup():
    """
    Simple test to ensure pytest is working.
    """
    assert 1 + 1 == 2

def test_pandas_import():
    """
    Test to ensure pandas is installed and working.
    """
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    assert not df.empty
    assert df.shape == (2, 2)

def test_numpy_operation():
    """
    Test to ensure numpy is installed and working.
    """
    arr = np.array([1, 2, 3])
    assert np.sum(arr) == 6