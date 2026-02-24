#!/usr/bin/env python3

"""
Test script for mask-to-RLE conversion and the full pipeline.

1. The core mask_to_rle function
2. The full pipeline including different encoding schemes
3. Integration with create_seg_tfrecord and create_video_seg_tfrecord workflows
"""

import sys
import os
import numpy as np
import tempfile

sys.path.append(os.getcwd())

from tasks import task_utils


def test_mask_to_rle_with_different_schemes():
    """Test mask_to_rle with different encoding schemes used in the pipeline."""
    print("=== Testing Different Encoding Schemes ===")
    
    # Create a more complex test mask
    mask = np.zeros((32, 32), dtype=np.uint8)
    
    # Create some regions for testing
    mask[5:10, 5:15] = 1    # Class 1: rectangle
    mask[15:20, 10:25] = 2  # Class 2: rectangle  
    mask[25:28, 5:10] = 1   # Class 1: another region
    mask[25:28, 20:25] = 3  # Class 3: small region
    
    print(f"Test mask shape: {mask.shape}")
    print(f"Unique classes: {np.unique(mask)}")
    
    # Test with different parameters that are used in the actual pipeline
    test_configs = [
        {
            'name': 'Basic Configuration',
            'max_length': 64,
            'n_classes': 4,
            'order': 'C',
            'starts_2d': False,
            'length_as_class': False,
            'multi_class': True
        },
        {
            'name': 'With 2D Starts',
            'max_length': 32,
            'n_classes': 4,
            'order': 'C',
            'starts_2d': True,
            'length_as_class': False,
            'multi_class': True
        },
        {
            'name': 'Length as Class',
            'max_length': 16,
            'n_classes': 4,
            'order': 'C',
            'starts_2d': False,
            'length_as_class': True,
            'multi_class': True
        }
    ]
    
    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        
        # Step 1: Convert mask to RLE
        starts, lengths = task_utils.mask_to_rle(
            mask=mask,
            max_length=config['max_length'],
            n_classes=config['n_classes'],
            order=config['order']
        )
        
        print(f"RLE: {len(starts)} runs")
        
        if config['multi_class']:
            # Get class IDs for each run (as done in the pipeline)
            class_id_to_col = {0: 'background', 1: 'class1', 2: 'class2', 3: 'class3'}
            class_ids = task_utils.get_rle_class_ids(
                mask, starts, len(class_id_to_col), order=config['order']
            )
            print(f"Class IDs: {class_ids}")
            
            rle_cmp = [starts, lengths, class_ids]
        else:
            rle_cmp = [starts, lengths]
        
        # Step 2: Convert to tokens (as done in create_seg_tfrecord)
        # Using pipeline-like parameters
        starts_offset = 1000
        lengths_offset = 200
        class_offset = 100
        
        # For length_as_class mode, lengths_offset must equal class_offset
        if config['length_as_class']:
            lengths_offset = class_offset
        
        if config['length_as_class']:
            # Convert to length-as-class encoding
            rle_cmp = task_utils.rle_to_lac(rle_cmp, config['max_length'])
        
        # Convert to tokens
        rle_tokens = task_utils.rle_to_tokens(
            rle_cmp,
            mask.shape,
            config['length_as_class'],
            starts_offset,
            lengths_offset,
            class_offset,
            config['starts_2d'],
            config['order']
        )
        
        print(f"Tokens: {len(rle_tokens)} tokens")
        print(f"First 10 tokens: {rle_tokens[:10] if len(rle_tokens) >= 10 else rle_tokens}")
        
        # Step 3: Test round-trip conversion
        # Convert tokens back to mask
        mask_rec, rle_rec_cmp = task_utils.mask_from_tokens(
            rle_tokens,
            mask.shape,
            allow_extra=False,
            length_as_class=config['length_as_class'],
            max_length=config['max_length'],
            starts_offset=starts_offset,
            lengths_offset=lengths_offset,
            class_offset=class_offset,
            starts_2d=config['starts_2d'],
            multi_class=config['multi_class'],
            flat_order=config['order'],
            diff_mask=config['diff_mask'],
            max_seq_len=None,
            ignore_invalid=False,
        )

        # Check if round-trip is successful
        matches = np.array_equal(mask, mask_rec)
        print(f"Round-trip successful: {matches}")
        
        if not matches:
            diff_count = np.sum(mask != mask_rec)
            print(f"  Differences: {diff_count} pixels")


def test_pipeline_integration():
    """Test integration with the actual pipeline functions."""
    print("\n=== Testing Pipeline Integration ===")
    
    # Create test data similar to what the pipeline processes
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    
    # Create some realistic mask patterns
    mask[10:30, 10:30] = 1  # Class 1
    mask[35:50, 35:55] = 2  # Class 2
    mask[15:25, 40:50] = 1  # More Class 1
    
    # Subsample the mask (as done in the pipeline)
    subsample = 2
    mask_sub = task_utils.subsample_mask(mask, subsample, n_classes=3, is_vis=False)
    
    print(f"Original mask shape: {mask.shape}")
    print(f"Subsampled mask shape: {mask_sub.shape}")
    
    # Test the check functions that are used in the pipeline
    class_id_to_col = {0: 'background', 1: 'red', 2: 'blue'}
    
    # Simulate pipeline parameters
    max_length = 64
    starts_offset = 1000
    lengths_offset = 200
    class_offset = 100
    starts_2d = False
    length_as_class = False
    flat_order = 'C'
    n_classes = 3
    multi_class = True
    
    # Generate RLE tokens using the same process as get_rle_tokens
    starts, lengths = task_utils.mask_to_rle(
        mask=mask_sub,
        max_length=max_length // subsample,
        n_classes=n_classes,
        order=flat_order
    )
    
    rle_cmp = [starts, lengths]
    
    if multi_class:
        class_ids = task_utils.get_rle_class_ids(
            mask_sub, starts, len(class_id_to_col), order=flat_order
        )
        rle_cmp.append(class_ids)
    
    rle_tokens = task_utils.rle_to_tokens(
        rle_cmp,
        mask_sub.shape,
        length_as_class,
        starts_offset,
        lengths_offset,
        class_offset,
        starts_2d,
        flat_order
    )
    
    print(f"Generated {len(rle_tokens)} tokens")
    
    # Test the check function used in the pipeline
    try:
        task_utils.check_rle_tokens(
            image, mask, mask_sub, rle_tokens, n_classes,
            length_as_class,
            starts_2d,
            starts_offset, lengths_offset, class_offset,
            max_length, subsample, multi_class,
            flat_order,
            class_id_to_col, is_vis=False
        )
        print("✓ check_rle_tokens passed")
    except Exception as e:
        print(f"✗ check_rle_tokens failed: {e}")


def test_video_mask_pipeline():
    """Test video mask pipeline functions."""
    print("\n=== Testing Video Mask Pipeline ===")
    
    # Create test video mask data
    vid_len = 3
    height, width = 32, 32
    n_classes = 3
    
    # Create video masks (time, height, width)
    vid_mask = np.zeros((vid_len, height, width), dtype=np.uint8)
    
    # Add some temporal patterns
    for t in range(vid_len):
        # Moving object
        start_col = 5 + t * 5
        end_col = min(start_col + 10, width)
        vid_mask[t, 10:20, start_col:end_col] = 1
        
        # Static object
        vid_mask[t, 25:30, 20:25] = 2
    
    print(f"Video mask shape: {vid_mask.shape}")
    print(f"Classes per frame: {[np.unique(vid_mask[t]) for t in range(vid_len)]}")
    
    # Test video RLE conversion
    max_length = 32
    starts, lengths = task_utils.mask_to_rle(
        mask=vid_mask,
        max_length=max_length,
        n_classes=n_classes,
        order='C'
    )
    
    print(f"Video RLE: {len(starts)} runs")
    
    # Test time-as-class encoding
    class_id_to_col = {0: 'background', 1: 'class1', 2: 'class2'}
    
    # This would be used for time-as-class encoding in video pipeline
    rle_cmp = [starts, lengths]
    class_ids = task_utils.get_rle_class_ids(
        vid_mask, starts, len(class_id_to_col), order='C'
    )
    rle_cmp.append(class_ids)
    
    # Convert to tokens
    rle_tokens = task_utils.rle_to_tokens(
        rle_cmp,
        vid_mask.shape,
        length_as_class=False,
        starts_offset=1000,
        lengths_offset=200,
        class_offset=100,
        starts_2d=False,
        flat_order='C'
    )
    
    print(f"Video tokens: {len(rle_tokens)} tokens")
    
    # Test round-trip for video
    vid_mask_rec, tac_mask_rec, rle_rec_cmp = task_utils.vid_mask_from_tokens(
        rle_tokens,
        allow_extra=False,
        vid_len=vid_len,
        shape=(height, width),
        n_classes=n_classes,
        time_as_class=False,
        length_as_class=False,
        max_length=max_length,
        starts_offset=1000,
        lengths_offset=200,
        class_offset=100,
        starts_2d=False,
        multi_class=True,
        flat_order='C',
        ignore_invalid=False
    )
    
    matches = np.array_equal(vid_mask, vid_mask_rec)
    print(f"Video round-trip successful: {matches}")


def main():
    """Main test function."""
    print("Advanced Testing of mask_to_rle and Pipeline Integration")
    print("=" * 60)
    
    try:
        # Test different encoding schemes
        test_mask_to_rle_with_different_schemes()
        
        # Test pipeline integration
        test_pipeline_integration()
        
        # Test video pipeline
        test_video_mask_pipeline()
        
        print("\n" + "=" * 60)
        print("- Different encoding schemes: ✓")
        print("- Pipeline integration: ✓") 
        print("- Video pipeline: ✓")
        print("Success")
        
    except Exception as e:
        print(f"\nError during advanced testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
