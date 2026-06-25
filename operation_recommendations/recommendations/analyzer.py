"""
Trend Analyzer Module
=====================

Provides the TrendAnalyzer class for computing trends and rolling statistics
from sensor data time series.

Usage:
    >>> from operation_recommendations.recommendations.analyzer import TrendAnalyzer
    >>> analyzer = TrendAnalyzer(df, lift_type='ESP')
    >>> metrics = analyzer.compute_metrics()
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from operation_recommendations.config import get_sensors_for_lift_type


class TrendAnalyzer:
    """
    Compute simple trends and rolling stats from a dataframe of readings.

    Dynamically computes metrics for all available sensor columns based on lift type.
    Expects a DataFrame indexed by timestamp with sensor columns.
    
    Attributes:
        df: DataFrame of sensor readings, sorted by index
        lift_type: Optional lift type for context-aware analysis
        sensor_columns: List of sensor columns to analyze
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'motor_temp': [150, 155, 160, 165, 170],
        ...     'motor_current': [50, 52, 54, 56, 58]
        ... }, index=pd.date_range('2024-01-01', periods=5, freq='H'))
        >>> 
        >>> analyzer = TrendAnalyzer(df, lift_type='ESP')
        >>> metrics = analyzer.compute_metrics()
        >>> print(f"Motor temp slope: {metrics.get('motor_temp_slope')}")
    """

    def __init__(self, df: pd.DataFrame, lift_type: Optional[str] = None):
        """
        Initialize the TrendAnalyzer.
        
        Args:
            df: DataFrame with sensor readings, should have a datetime index
            lift_type: Optional lift type (Rod Pump, ESP, Gas Lift) for context
        """
        self.df = df.sort_index()
        self.lift_type = lift_type
        
        # Get sensor columns for this lift type (or all numeric columns if unknown)
        if lift_type:
            self.sensor_columns = [
                c for c in get_sensors_for_lift_type(lift_type) 
                if c in df.columns
            ]
        else:
            # Use all numeric columns
            self.sensor_columns = list(
                df.select_dtypes(include=[np.number]).columns
            )

    def rolling_means(self, window: str = '24H') -> pd.DataFrame:
        """
        Compute rolling means for all columns.
        
        Args:
            window: Rolling window size (pandas offset string)
            
        Returns:
            DataFrame with rolling means
        """
        return self.df.rolling(window).mean()

    def slope(self, series: pd.Series, window_points: int = 24) -> Optional[float]:
        """
        Compute slope over the last `window_points` samples using linear fit.
        
        Args:
            series: Time series data
            window_points: Number of points to use for slope calculation
            
        Returns:
            Slope value or None if insufficient data
        """
        y = series.dropna()
        if len(y) < max(3, window_points):
            return None
        y_tail = y.iloc[-window_points:]
        x = np.arange(len(y_tail))
        try:
            m = np.polyfit(x, y_tail.values, 1)[0]
            return float(m)
        except Exception:
            return None

    def baseline_stats(
        self, 
        field: str, 
        lookback_points: int = 720, 
        exclude_last: int = 24
    ) -> Dict[str, Optional[float]]:
        """
        Compute baseline mean/std and slope for a field using a lookback window.

        Args:
            field: Column name to analyze
            lookback_points: Number of historical samples to use (e.g., 720 for 30 days hourly)
            exclude_last: Number of most recent samples to exclude from baseline (e.g., 24 hours)
            
        Returns:
            Dict with keys: mean, std, slope, slope_std
        """
        if field not in self.df.columns:
            return {'mean': None, 'std': None, 'slope': None, 'slope_std': None}
        
        series = self.df[field].dropna()
        if len(series) < (exclude_last + 3):
            return {'mean': None, 'std': None, 'slope': None, 'slope_std': None}
        
        end_idx = len(series) - exclude_last
        start_idx = max(0, end_idx - lookback_points)
        window = series.iloc[start_idx:end_idx]
        
        mean = float(window.mean()) if not window.empty else None
        std = float(window.std()) if not window.empty else None
        
        # Compute slope over the baseline window
        slope = None
        slope_std = None
        try:
            if len(window) >= 3:
                x = np.arange(len(window))
                m = np.polyfit(x, window.values, 1)[0]
                slope = float(m)
                
                # Approximate slope std by sampling smaller windows
                sls = []
                w = 24
                for i in range(0, max(1, len(window) - w), max(1, w)):
                    part = window.iloc[i:i + w]
                    if len(part) >= 3:
                        xm = np.arange(len(part))
                        sls.append(np.polyfit(xm, part.values, 1)[0])
                slope_std = float(np.std(sls)) if sls else 0.0
        except Exception:
            slope = None
            slope_std = None

        return {'mean': mean, 'std': std, 'slope': slope, 'slope_std': slope_std}

    def compute_metrics(self, include_all_sensors: bool = True) -> Dict[str, Any]:
        """
        Compute metrics for all available sensor columns.
        
        Args:
            include_all_sensors: If True, compute metrics for ALL numeric columns.
                               If False, only compute for lift-type specific sensors.
        
        Returns:
            Dict of metrics with {sensor}_slope, {sensor}_roll_mean, and {sensor}_latest
            for each sensor, plus missing_rate and lift-type specific correlations.
        """
        metrics: Dict[str, Any] = {}
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Determine which columns to process
        if include_all_sensors:
            sensor_cols = list(numeric_df.columns)
        else:
            sensor_cols = self.sensor_columns
        
        # Compute slopes for all sensor columns
        for col in sensor_cols:
            if col in self.df.columns:
                metrics[f'{col}_slope'] = self.slope(self.df[col], window_points=24)
            else:
                metrics[f'{col}_slope'] = None

        # Compute rolling means for all sensor columns
        rm = (
            numeric_df.rolling(window=24).mean().iloc[-1] 
            if len(numeric_df) >= 24 
            else numeric_df.mean()
        )
        for col in sensor_cols:
            if col in rm.index and not pd.isna(rm[col]):
                metrics[f'{col}_roll_mean'] = float(rm[col])
            else:
                metrics[f'{col}_roll_mean'] = None
        
        # Store latest values for each sensor (for rule checking)
        if len(self.df) > 0:
            last_row = self.df.iloc[-1]
            for col in sensor_cols:
                if col in last_row.index:
                    val = last_row[col]
                    metrics[f'{col}_latest'] = float(val) if pd.notna(val) else None

        # Missing rate (data quality indicator)
        metrics['missing_rate'] = float(self.df.isna().mean().mean())

        # Compute correlations based on lift type
        self._add_correlations(metrics)

        return metrics

    def _add_correlations(self, metrics: Dict[str, Any]) -> None:
        """
        Add lift-type specific correlations to metrics dict.
        
        Args:
            metrics: Dict to add correlations to (modified in place)
        """
        if self.lift_type == 'ESP':
            self._add_esp_correlations(metrics)
        elif self.lift_type == 'Rod Pump':
            self._add_rod_pump_correlations(metrics)
        elif self.lift_type == 'Gas Lift':
            self._add_gas_lift_correlations(metrics)
        else:
            # Default correlation (legacy support)
            if 'motor_current' in self.df.columns and 'motor_temp' in self.df.columns:
                corr = self.df['motor_current'].corr(self.df['motor_temp'])
                metrics['motor_current_motor_temp_corr'] = float(corr) if pd.notna(corr) else None

    def _add_esp_correlations(self, metrics: Dict[str, Any]) -> None:
        """Add ESP-specific correlations."""
        if 'motor_current' in self.df.columns and 'motor_temp' in self.df.columns:
            corr = self.df['motor_current'].corr(self.df['motor_temp'])
            metrics['motor_current_motor_temp_corr'] = float(corr) if pd.notna(corr) else None

        vib_col = 'vibration_x' if 'vibration_x' in self.df.columns else None
        if vib_col and 'motor_current' in self.df.columns:
            corr = self.df['motor_current'].corr(self.df[vib_col])
            metrics['motor_current_vibration_corr'] = float(corr) if pd.notna(corr) else None

        freq_col = 'freq_hz' if 'freq_hz' in self.df.columns else 'drive_frequency'
        if 'motor_current' in self.df.columns and freq_col in self.df.columns:
            corr = self.df['motor_current'].corr(self.df[freq_col])
            metrics['motor_current_freq_corr'] = float(corr) if pd.notna(corr) else None

        if 'motor_load' in self.df.columns and freq_col in self.df.columns:
            corr = self.df['motor_load'].corr(self.df[freq_col])
            metrics['load_frequency_corr'] = float(corr) if pd.notna(corr) else None

    def _add_rod_pump_correlations(self, metrics: Dict[str, Any]) -> None:
        """Add Rod Pump-specific correlations."""
        if 'strokes_per_minute' in self.df.columns and 'polish_rod_load' in self.df.columns:
            corr = self.df['strokes_per_minute'].corr(self.df['polish_rod_load'])
            metrics['spm_load_corr'] = float(corr) if pd.notna(corr) else None
        elif 'strokes_per_minute' in self.df.columns and 'motor_current' in self.df.columns:
            corr = self.df['strokes_per_minute'].corr(self.df['motor_current'])
            metrics['spm_load_corr'] = float(corr) if pd.notna(corr) else None

        if 'pump_fillage' in self.df.columns and 'inferred_production' in self.df.columns:
            corr = self.df['pump_fillage'].corr(self.df['inferred_production'])
            metrics['fillage_production_corr'] = float(corr) if pd.notna(corr) else None
        elif 'pump_fillage' in self.df.columns and 'tubing_pressure' in self.df.columns:
            corr = self.df['pump_fillage'].corr(self.df['tubing_pressure'])
            metrics['fillage_production_corr'] = float(corr) if pd.notna(corr) else None

        if 'strokes_per_minute' in self.df.columns and 'torque' in self.df.columns:
            corr = self.df['strokes_per_minute'].corr(self.df['torque'])
            metrics['spm_torque_corr'] = float(corr) if pd.notna(corr) else None

    def _add_gas_lift_correlations(self, metrics: Dict[str, Any]) -> None:
        """Add Gas Lift-specific correlations."""
        pressure_col = (
            'bottomhole_pressure'
            if 'bottomhole_pressure' in self.df.columns
            else 'casing_pressure'
        )
        if 'injection_rate' in self.df.columns and pressure_col in self.df.columns:
            corr = self.df['injection_rate'].corr(self.df[pressure_col])
            metrics['injection_production_corr'] = float(corr) if pd.notna(corr) else None

        if 'injection_pressure' in self.df.columns and 'injection_rate' in self.df.columns:
            corr = self.df['injection_pressure'].corr(self.df['injection_rate'])
            metrics['pressure_rate_corr'] = float(corr) if pd.notna(corr) else None

        if 'plunger_velocity' in self.df.columns and 'arrival_count' in self.df.columns:
            corr = self.df['plunger_velocity'].corr(self.df['arrival_count'])
            metrics['velocity_arrival_corr'] = float(corr) if pd.notna(corr) else None
