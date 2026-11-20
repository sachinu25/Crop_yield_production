import pandas as pd
import pytest

from src.services.db_builder import validate_data


def test_validate_data(sample_raw_dataframe):
    """Test that data validation drops nulls and negative areas."""
    initial_len = len(sample_raw_dataframe)
    
    # The sample has 3 rows:
    # 1. Valid row
    # 2. Negative area (-500.0) -> should be dropped
    # 3. Null state_name -> should be dropped
    
    clean_df = validate_data(sample_raw_dataframe)
    
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["state_name"] == "Punjab"
    assert clean_df.iloc[0]["area"] == 1000.5
