import pytest
import pandas as pd
import numpy as np

def _get_df(sz, freq='H'):
    df = pd.DataFrame(
        data = {
            'y': np.arange(sz),
            'x': np.arange(sz)*100,
            },
        index = pd.date_range('2013-01-01', periods=sz, freq=freq))
    return df

@pytest.fixture(scope='module', params = [
    (1000, 'H'),
  #  (100, 'H'), # TODO: check out of boundaries
  #  (int(1e6), 'H'), # TODO: check out of memory, inefficient copying, etc
])
def df(request):
    return _get_df(*request.param)

@pytest.fixture(scope='module')
def horiz():
    return 20

@pytest.fixture(scope='module')
def target():
    return 'y'