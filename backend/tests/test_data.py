import pytest
import pandas as pd
from app.data.synthetic import SyntheticDataHandler

def test_synthetic_data_generation():
    handler = SyntheticDataHandler(seed=42)
    df = handler.fetch_data("TEST", "2023-01-01", "2023-01-10", "1d")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 10

    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in expected_cols:
        assert col in df.columns

    # Check logic: High >= Low, High >= Open, High >= Close
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
    assert (df['low'] <= df['open']).all()
    assert (df['low'] <= df['close']).all()

def test_synthetic_data_reproducibility():
    h1 = SyntheticDataHandler(seed=42)
    df1 = h1.fetch_data("TEST", "2023-01-01", "2023-01-10")

    h2 = SyntheticDataHandler(seed=42)
    df2 = h2.fetch_data("TEST", "2023-01-01", "2023-01-10")

    pd.testing.assert_frame_equal(df1, df2)
