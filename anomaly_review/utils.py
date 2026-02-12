"""
Utility functions for anomaly review.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('AnomalyReviewUtils')

def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, handling zero division and NaNs.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Return value if division is invalid
        
    Returns:
        Result of division or default value
    """
    try:
        if pd.isna(numerator) or pd.isna(denominator):
            return default
            
        if denominator == 0:
            return default
            
        return float(numerator) / float(denominator)
    except Exception:
        return default
