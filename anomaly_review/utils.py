
from typing import Union, Optional
import numpy as np
import pandas as pd

def safe_divide(numerator: Union[float, int, pd.Series, np.ndarray], 
                denominator: Union[float, int, pd.Series, np.ndarray], 
                default: float = 0.0) -> Union[float, pd.Series, np.ndarray]:
    """
    Safely divide numerator by denominator, returning default if denominator is zero (or close to it).
    Handles scalars, pandas Series, and numpy arrays.
    """
    try:
        # Handle pandas Series/DF
        if isinstance(numerator, (pd.Series, pd.DataFrame)) or isinstance(denominator, (pd.Series, pd.DataFrame)):
            # Determine result based on pandas alignment
            # Using numpy `where` for element-wise condition on series requires aligning them first if they are series
            # But straightforward division with fillna is often easiest for Series.
            
            # Simple division
            result = numerator / denominator
            
            # Replace infinities and NaNs with default
            result = result.replace([np.inf, -np.inf], np.nan).fillna(default)
            return result
            
        # Handle numpy arrays
        elif isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.true_divide(numerator, denominator)
                result[~np.isfinite(result)] = default
                return result
                
        # Handle scalars
        else:
            if denominator == 0:
                return default
            return numerator / denominator
            
    except Exception:
        return default
