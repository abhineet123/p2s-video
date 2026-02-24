#!/usr/bin/env python3
"""
Test script for graph-mode compatible mask_to_rle function.
Tests both eager mode and graph mode TensorFlow implementations.
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())

from tasks import task_utils


def test_mask_to_rle_graph_mode():
    """Test the graph-mode compatible TensorFlow implementation."""
    print("Testing mask_to_rle_graph_mode function...")

    # Test with simple binary mask
    mask = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [2, 2, 0, 0]
    ], dtype=np.uint8)

    mask_tf = tf.constant(mask)

    # Test in graph mode
    @tf.function
    def test_graph_execution():
        return task_utils.mask_to_rle_graph_mode(
            mask_tf, max_length=10000, n_classes=3, order='C', randomize_runs=False)

    starts_graph, lengths_graph = test_graph_execution()

    print(f"Graph mode - Starts: {starts_graph.numpy()}")
    print(f"Graph mode - Lengths: {lengths_graph.numpy()}")

    # Compare with original numpy implementation
    starts_np, lengths_np = task_utils.mask_to_rle(mask, max_length=10000, n_classes=3, order='C')

    print(f"NumPy - Starts: {starts_np}")
    print(f"NumPy - Lengths: {lengths_np}")

    # Verify they match
    assert np.array_equal(starts_graph.numpy(), starts_np), f"Starts mismatch: {starts_graph.numpy()} vs {starts_np}"
    assert np.array_equal(lengths_graph.numpy(),
                          lengths_np), f"Lengths mismatch: {lengths_graph.numpy()} vs {lengths_np}"

    print("✓ Graph mode test passed!")
    return True


def test_eager_vs_graph_mode():
    """Compare eager mode and graph mode implementations."""
    print("\nTesting eager vs graph mode consistency...")

    # Create a more complex test mask
    mask = np.array([
        [0, 1, 1, 2, 2],
        [1, 1, 0, 2, 0],
        [0, 0, 1, 1, 1],
        [2, 0, 0, 1, 2]
    ], dtype=np.uint8)

    mask_tf = tf.constant(mask)

    # Test eager mode (existing function)
    starts_eager, lengths_eager = task_utils.mask_to_rle_tf(mask_tf, max_length=10000, n_classes=3, order='C')

    # Test graph mode
    @tf.function
    def graph_test():
        return task_utils.mask_to_rle_graph_mode(
            mask_tf, max_length=10000, n_classes=3, order='C', randomize_runs=False)

    starts_graph, lengths_graph = graph_test()

    print(f"Eager mode - Starts: {starts_eager.numpy()}")
    print(f"Eager mode - Lengths: {lengths_eager.numpy()}")
    print(f"Graph mode - Starts: {starts_graph.numpy()}")
    print(f"Graph mode - Lengths: {lengths_graph.numpy()}")

    # They should produce the same results
    assert np.array_equal(starts_eager.numpy(), starts_graph.numpy()), "Starts don't match between eager and graph mode"
    assert np.array_equal(lengths_eager.numpy(),
                          lengths_graph.numpy()), "Lengths don't match between eager and graph mode"

    print("✓ Eager vs graph mode consistency test passed!")
    return True


def test_max_length_splitting_graph_mode():
    """Test that max_length splitting works in graph mode."""
    print("\nTesting max_length splitting in graph mode...")

    # Create mask with long runs that need splitting
    mask = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],  # 8 consecutive 1s
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    mask_tf = tf.constant(mask)
    max_length = 3

    @tf.function
    def test_splitting():
        return task_utils.mask_to_rle_graph_mode(
            mask_tf, max_length=max_length, n_classes=2, order='C', randomize_runs=False)

    starts, lengths = test_splitting()

    print(f"Starts: {starts.numpy()}")
    print(f"Lengths: {lengths.numpy()}")

    # Verify no length exceeds max_length
    assert np.all(lengths.numpy() <= max_length), f"Some lengths exceed max_length: {lengths.numpy()}"

    # Verify total coverage is correct
    total_pixels = np.sum(lengths.numpy())
    expected_pixels = 8  # 8 consecutive 1s
    assert total_pixels == expected_pixels, f"Total pixels mismatch: {total_pixels} vs {expected_pixels}"

    print("✓ Max length splitting test passed!")
    return True


def test_all_implementations():
    """Test all three implementations for consistency."""
    print("\nTesting all implementations for consistency...")

    # Create test mask
    mask = np.array([
        [0, 1, 1, 0, 2],
        [1, 0, 1, 2, 2],
        [0, 0, 0, 1, 0]
    ], dtype=np.uint8)

    mask_tf = tf.constant(mask)

    # NumPy implementation
    starts_np, lengths_np = task_utils.mask_to_rle(mask, max_length=10000, n_classes=3, order='C')

    # TensorFlow eager mode
    starts_eager, lengths_eager = task_utils.mask_to_rle_tf(mask_tf, max_length=10000, n_classes=3, order='C')

    # TensorFlow graph mode
    @tf.function
    def graph_implementation():
        return task_utils.mask_to_rle_graph_mode(
            mask_tf, max_length=10000,
            n_classes=3, order='C', randomize_runs=False)

    starts_graph, lengths_graph = graph_implementation()

    print(f"NumPy:      Starts={starts_np}, Lengths={lengths_np}")
    print(f"TF Eager:   Starts={starts_eager.numpy()}, Lengths={lengths_eager.numpy()}")
    print(f"TF Graph:   Starts={starts_graph.numpy()}, Lengths={lengths_graph.numpy()}")

    # All should match
    assert np.array_equal(starts_np, starts_eager.numpy()), "NumPy vs Eager mismatch"
    assert np.array_equal(starts_np, starts_graph.numpy()), "NumPy vs Graph mismatch"
    assert np.array_equal(lengths_np, lengths_eager.numpy()), "NumPy vs Eager lengths mismatch"
    assert np.array_equal(lengths_np, lengths_graph.numpy()), "NumPy vs Graph lengths mismatch"

    print("✓ All implementations produce consistent results!")
    return True


def test_empty_mask():
    """Test with empty mask (all zeros)."""
    print("\nTesting empty mask...")

    mask = np.zeros((3, 3), dtype=np.uint8)
    mask_tf = tf.constant(mask)

    @tf.function
    def test_empty():
        return task_utils.mask_to_rle_graph_mode(
            mask_tf, max_length=10000, n_classes=2, order='C', randomize_runs=False)

    starts, lengths = test_empty()

    print(f"Empty mask - Starts: {starts.numpy()}")
    print(f"Empty mask - Lengths: {lengths.numpy()}")

    # Should be empty arrays
    assert len(starts.numpy()) == 0, "Empty mask should produce empty starts"
    assert len(lengths.numpy()) == 0, "Empty mask should produce empty lengths"

    print("✓ Empty mask test passed!")
    return True


def test_fortran_order():
    """Test with Fortran order flattening."""
    print("\nTesting Fortran order...")

    mask = np.array([
        [1, 0, 2],
        [1, 2, 0],
        [0, 2, 1]
    ], dtype=np.uint8)

    mask_tf = tf.constant(mask)

    # Test both C and F order
    @tf.function
    def test_c_order():
        return task_utils.mask_to_rle_graph_mode(
            mask_tf, max_length=10000, n_classes=3, order='C', randomize_runs=False)

    @tf.function
    def test_f_order():
        return task_utils.mask_to_rle_graph_mode(
            mask_tf, max_length=10000, n_classes=3, order='F', randomize_runs=False)

    starts_c, lengths_c = test_c_order()
    starts_f, lengths_f = test_f_order()

    print(f"C order - Starts: {starts_c.numpy()}, Lengths: {lengths_c.numpy()}")
    print(f"F order - Starts: {starts_f.numpy()}, Lengths: {lengths_f.numpy()}")

    # Compare with numpy implementation
    starts_np_c, lengths_np_c = task_utils.mask_to_rle(mask, max_length=10000, n_classes=3, order='C')
    starts_np_f, lengths_np_f = task_utils.mask_to_rle(mask, max_length=10000, n_classes=3, order='F')

    assert np.array_equal(starts_c.numpy(), starts_np_c), "C order starts mismatch"
    assert np.array_equal(lengths_c.numpy(), lengths_np_c), "C order lengths mismatch"
    assert np.array_equal(starts_f.numpy(), starts_np_f), "F order starts mismatch"
    assert np.array_equal(lengths_f.numpy(), lengths_np_f), "F order lengths mismatch"

    print("✓ Fortran order test passed!")
    return True


def main():
    """Run all tests."""
    print("Testing TensorFlow graph-mode compatible mask_to_rle implementation")
    print("=" * 70)

    try:
        test_mask_to_rle_graph_mode()
        test_eager_vs_graph_mode()
        test_max_length_splitting_graph_mode()
        test_all_implementations()
        test_empty_mask()
        test_fortran_order()

        print("\n" + "=" * 70)
        print("All graph mode tests passed successfully!")
        print("The graph-mode implementation is ready for training pipelines.")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
