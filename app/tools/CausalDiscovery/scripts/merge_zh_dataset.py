"""
CSV Merging Script for zh_dataset/0922

This script merges CSV files in the zh_dataset/0922 directory with the following requirements:
1. Merge normal and anomalous CSV files by column (same metrics, different time periods)
2. Convert 10-digit timestamps to Beijing time format (2025-09-22 02:50:00)
3. Convert UTC time in addition.csv to Beijing time and align with merged file
4. Forward fill missing values

Usage:
    python merge_zh_dataset.py --input_dir data/raw/zh_dataset/0922 --output_dir data/processed/zh_dataset/0922
"""

import argparse
import os
import sys
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZhDatasetMerger:
    """Class for merging zh_dataset CSV files"""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the merger
        
        Args:
            input_dir: Input directory containing CSV files
            output_dir: Output directory for merged files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized merger for input: {input_dir}, output: {output_dir}")
    
    def _convert_timestamp_to_beijing(self, timestamp: int) -> str:
        """
        Convert 10-digit timestamp to Beijing time string
        
        Args:
            timestamp: 10-digit Unix timestamp
            
        Returns:
            Beijing time string in format 'YYYY-MM-DD HH:MM:SS'
        """
        try:
            # Convert timestamp to UTC datetime
            utc_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            
            # Convert to Beijing time (UTC+8)
            beijing_dt = utc_dt + timedelta(hours=8)
            
            # Format as string
            return beijing_dt.strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            logger.error(f"Failed to convert timestamp {timestamp}: {e}")
            return str(timestamp)
    
    def _convert_utc_to_beijing(self, utc_str: str) -> str:
        """
        Convert UTC time string to Beijing time string
        
        Args:
            utc_str: UTC time string (various formats supported)
            
        Returns:
            Beijing time string in format 'YYYY-MM-DD HH:MM:SS'
        """
        try:
            # Try different time formats
            time_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S%z',
                '%Y-%m-%d %H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%fZ',
            ]
            
            dt = None
            for fmt in time_formats:
                try:
                    dt = datetime.strptime(utc_str, fmt)
                    break
                except ValueError:
                    continue
            
            if dt is None:
                raise ValueError(f"Unable to parse time string: {utc_str}")
            
            # If no timezone info, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            # Convert to Beijing time (UTC+8)
            beijing_dt = dt + timedelta(hours=8)
            
            # Format as string
            return beijing_dt.strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            logger.error(f"Failed to convert UTC time {utc_str}: {e}")
            return utc_str
    
    def _load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from input directory
        
        Returns:
            Dictionary mapping filename to DataFrame
        """
        csv_files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")
        
        dataframes = {}
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            logger.info(f"Loading {filename}")
            
            try:
                df = pd.read_csv(csv_file)
                dataframes[filename] = df
                logger.info(f"Loaded {filename}: {df.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                continue
        
        return dataframes
    
    def _merge_normal_anomalous(self, normal_df: pd.DataFrame, anomalous_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge normal and anomalous DataFrames by column
        
        Args:
            normal_df: Normal period DataFrame
            anomalous_df: Anomalous period DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging normal and anomalous DataFrames")
        
        # Get column names (excluding time column)
        normal_cols = set(normal_df.columns)
        anomalous_cols = set(anomalous_df.columns)
        
        # Check if they have the same metrics (excluding time)
        time_col_normal = None
        time_col_anomalous = None
        
        # Find time columns
        for col in normal_df.columns:
            if col.lower() in ['time', 'timestamp']:
                time_col_normal = col
                break
        
        for col in anomalous_df.columns:
            if col.lower() in ['time', 'timestamp']:
                time_col_anomalous = col
                break
        
        if time_col_normal:
            normal_cols.discard(time_col_normal)
        if time_col_anomalous:
            anomalous_cols.discard(time_col_anomalous)
        
        # Check metric overlap
        common_metrics = normal_cols & anomalous_cols
        normal_only = normal_cols - anomalous_cols
        anomalous_only = anomalous_cols - normal_cols
        
        logger.info(f"Common metrics: {len(common_metrics)}")
        logger.info(f"Normal only metrics: {len(normal_only)}")
        logger.info(f"Anomalous only metrics: {len(anomalous_only)}")
        
        if common_metrics:
            logger.info(f"Common metrics: {list(common_metrics)}")
        
        # Merge DataFrames
        if time_col_normal and time_col_anomalous:
            # Both have time columns, concatenate vertically
            merged_df = pd.concat([normal_df, anomalous_df], ignore_index=True)
            logger.info(f"Merged vertically: {merged_df.shape}")
        else:
            # Concatenate horizontally
            merged_df = pd.concat([normal_df, anomalous_df], axis=1)
            logger.info(f"Merged horizontally: {merged_df.shape}")
        
        return merged_df
    
    def _process_time_columns(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Process time columns based on file type
        
        Args:
            df: DataFrame to process
            filename: Name of the source file
            
        Returns:
            DataFrame with processed time columns
        """
        df_processed = df.copy()
        
        # Find time column
        time_col = None
        for col in df.columns:
            if col.lower() in ['time', 'timestamp']:
                time_col = col
                break
        
        if not time_col:
            logger.warning(f"No time column found in {filename}")
            return df_processed
        
        logger.info(f"Processing time column '{time_col}' in {filename}")
        
        if 'addition' in filename.lower():
            # Convert UTC to Beijing time
            df_processed[time_col] = df_processed[time_col].apply(self._convert_utc_to_beijing)
            logger.info("Converted UTC time to Beijing time")
        else:
            # Convert 10-digit timestamp to Beijing time
            df_processed[time_col] = df_processed[time_col].apply(
                lambda x: self._convert_timestamp_to_beijing(int(x)) if pd.notna(x) else x
            )
            logger.info("Converted 10-digit timestamps to Beijing time")
        
        return df_processed
    
    def _align_time_series(self, main_df: pd.DataFrame, addition_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align time series between main DataFrame and addition DataFrame
        
        Args:
            main_df: Main DataFrame (merged normal+anomalous)
            addition_df: Addition DataFrame
            
        Returns:
            Aligned and merged DataFrame
        """
        logger.info("Aligning time series")
        
        # Find time columns
        main_time_col = None
        addition_time_col = None
        
        for col in main_df.columns:
            if col.lower() in ['time', 'timestamp']:
                main_time_col = col
                break
        
        for col in addition_df.columns:
            if col.lower() in ['time', 'timestamp']:
                addition_time_col = col
                break
        
        if not main_time_col or not addition_time_col:
            raise ValueError("Time columns not found in DataFrames")
        
        # Convert time columns to datetime for proper alignment
        main_df[main_time_col] = pd.to_datetime(main_df[main_time_col])
        addition_df[addition_time_col] = pd.to_datetime(addition_df[addition_time_col])
        
        # Sort by time
        main_df = main_df.sort_values(main_time_col)
        addition_df = addition_df.sort_values(addition_time_col)
        
        logger.info(f"Main DataFrame time range: {main_df[main_time_col].min()} to {main_df[main_time_col].max()}")
        logger.info(f"Addition DataFrame time range: {addition_df[addition_time_col].min()} to {addition_df[addition_time_col].max()}")
        
        # Merge on time column
        print(main_df, addition_df)
        merged_df = pd.merge(
            main_df,
            addition_df,
            on=main_time_col,
            how='outer',
            suffixes=('', '_addition')
        )
        
        # If time columns have different names, keep the main one
        if main_time_col != addition_time_col:
            merged_df = merged_df.drop(columns=[addition_time_col])
        
        # Sort by time
        merged_df = merged_df.sort_values(main_time_col)
        
        logger.info(f"Aligned DataFrame shape: {merged_df.shape}")
        
        return merged_df
    
    def _forward_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill missing values
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with forward-filled values
        """
        logger.info("Forward filling missing values")
        
        # Get non-time columns
        time_col = None
        for col in df.columns:
            if col.lower() in ['time', 'timestamp']:
                time_col = col
                break
        
        # Select numeric columns for forward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if time_col and time_col in numeric_cols:
            numeric_cols.remove(time_col)
        
        logger.info(f"Forward filling {len(numeric_cols)} numeric columns")
        
        # Forward fill
        df_filled = df.copy()
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(method='ffill')
        
        # Report missing values before and after
        missing_before = df[numeric_cols].isnull().sum().sum()
        missing_after = df_filled[numeric_cols].isnull().sum().sum()
        
        logger.info(f"Missing values before: {missing_before}, after: {missing_after}")
        
        return df_filled
    
    def merge_dataset(self) -> str:
        """
        Main method to merge the dataset
        
        Returns:
            Path to the merged output file
        """
        logger.info("Starting dataset merge process")
        
        # Load all CSV files
        dataframes = self._load_csv_files()
        
        # Find required files
        normal_file = None
        anomalous_file = None
        addition_file = None
        
        for filename in dataframes.keys():
            filename_lower = filename.lower()
            if 'normal' in filename_lower:
                normal_file = filename
            elif 'anomalous' in filename_lower:
                anomalous_file = filename
            elif 'addition' in filename_lower:
                addition_file = filename
        
        # Validate required files
        if not normal_file:
            raise FileNotFoundError("Normal CSV file not found")
        if not anomalous_file:
            raise FileNotFoundError("Anomalous CSV file not found")
        if not addition_file:
            raise FileNotFoundError("Addition CSV file not found")
        
        logger.info(f"Found files: normal={normal_file}, anomalous={anomalous_file}, addition={addition_file}")
        
        # Process time columns
        normal_df = self._process_time_columns(dataframes[normal_file], normal_file)
        anomalous_df = self._process_time_columns(dataframes[anomalous_file], anomalous_file)
        addition_df = self._process_time_columns(dataframes[addition_file], addition_file)
        
        # Merge normal and anomalous
        merged_normal_anomalous = self._merge_normal_anomalous(normal_df, anomalous_df)
        
        # Align with addition
        aligned_df = self._align_time_series(merged_normal_anomalous, addition_df)
        
        # Forward fill missing values
        final_df = self._forward_fill_missing(aligned_df)
        
        # Save output
        output_file = os.path.join(self.output_dir, "merged_dataset.csv")
        final_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved merged dataset to: {output_file}")
        logger.info(f"Final dataset shape: {final_df.shape}")
        logger.info(f"Final dataset columns: {list(final_df.columns)}")
        
        return output_file


def main():
    """Main function"""
    # Configuration variables - modify these values as needed
    input_dir = "data/raw/zh_dataset/0922"
    output_dir = "data/processed/zh_dataset/0922"
    verbose = True

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input directory
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create merger and run
    try:
        merger = ZhDatasetMerger(input_dir, output_dir)
        output_file = merger.merge_dataset()

        logger.info("Dataset merge completed successfully")
        logger.info(f"Output file: {output_file}")

    except Exception as e:
        logger.error(f"Dataset merge failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import numpy as np  # Import here to avoid circular imports
    main()