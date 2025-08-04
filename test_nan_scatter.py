#!/usr/bin/env python3
"""Test script to verify NaN handling in plot_scatter function."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.visual_insights.scatter import plot_scatter

def test_nan_handling():
    """Test that plot_scatter handles NaN values correctly."""
    print("Testing NaN handling in plot_scatter...")
    
    # Create test data with NaN values
    input_data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    target_data = np.array([2.0, np.nan, 4.0, 8.0, 10.0, 12.0])
    
    print(f"Input data: {input_data}")
    print(f"Target data: {target_data}")
    
    try:
        # This should work and filter out NaN values
        fig = plot_scatter(
            input_data, 
            target_data, 
            feature_name="Test Feature",
            target_name="Test Target",
            show=False
        )
        
        # Check that the figure was created
        assert fig is not None
        print("✓ NaN handling test passed - figure created successfully")
        
        # Test with all NaN values
        all_nan_input = np.array([np.nan, np.nan, np.nan])
        all_nan_target = np.array([np.nan, np.nan, np.nan])
        
        try:
            plot_scatter(all_nan_input, all_nan_target, show=False)
            print("✗ All NaN test failed - should have raised ValueError")
        except ValueError as e:
            print(f"✓ All NaN test passed - correctly raised: {e}")
            
    except Exception as e:
        print(f"✗ NaN handling test failed: {e}")
        raise

def test_normal_operation():
    """Test normal operation without NaN values."""
    print("\nTesting normal operation...")
    
    # Create clean test data
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target_data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    try:
        fig = plot_scatter(
            input_data,
            target_data,
            feature_name="Clean Feature",
            target_name="Clean Target",
            show=False
        )
        
        assert fig is not None
        print("✓ Normal operation test passed")
        
    except Exception as e:
        print(f"✗ Normal operation test failed: {e}")
        raise

if __name__ == "__main__":
    test_nan_handling()
    test_normal_operation()
    print("\nAll tests completed!")
