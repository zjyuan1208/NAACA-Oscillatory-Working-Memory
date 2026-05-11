"""
Minimal test to verify River library drift detection works at all
"""

import numpy as np
from river import drift

def test_adwin_basic():
    """Test ADWIN with the most obvious change possible"""
    print("="*50)
    print("TESTING ADWIN WITH OBVIOUS STEP CHANGE")
    print("="*50)

    # Create extremely obvious change: 0 -> 100
    data = []
    for i in range(100):
        if i < 50:
            data.append(0.0)  # Constant zero
        else:
            data.append(100.0)  # Constant 100

    print(f"Data: first 50 values = 0.0, next 50 values = 100.0")

    # Test different delta values
    deltas = [0.1, 0.01, 0.001, 0.0001, 0.00001]

    for delta in deltas:
        print(f"\nTesting delta = {delta}")
        detector = drift.ADWIN(delta=delta)

        detections = []
        for i, value in enumerate(data):
            detector.update(value)
            if detector.drift_detected:
                detections.append(i)
                print(f"  Change detected at step {i}")

        print(f"  Total detections: {len(detections)}")

        if detections:
            print(f"  SUCCESS: ADWIN can detect obvious changes")
            return True
        else:
            print(f"  FAILED: No detection with delta={delta}")

    print("\nADWIN COMPLETELY FAILED - even with most obvious change!")
    return False


def test_page_hinkley_basic():
    """Test Page-Hinkley with obvious change"""
    print("\n" + "="*50)
    print("TESTING PAGE-HINKLEY WITH OBVIOUS STEP CHANGE")
    print("="*50)

    # Same obvious data
    data = []
    for i in range(100):
        if i < 50:
            data.append(0.0)
        else:
            data.append(100.0)

    thresholds = [1.0, 5.0, 10.0, 50.0, 100.0]

    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold}")
        detector = drift.PageHinkley(
            min_instances=5,
            delta=0.01,
            threshold=threshold
        )

        detections = []
        for i, value in enumerate(data):
            detector.update(value)
            if detector.drift_detected:
                detections.append(i)
                print(f"  Change detected at step {i}")

        print(f"  Total detections: {len(detections)}")

        if detections:
            print(f"  SUCCESS: Page-Hinkley can detect obvious changes")
            return True

    print("\nPAGE-HINKLEY COMPLETELY FAILED!")
    return False


def test_kswin_basic():
    """Test KSWIN with obvious change"""
    print("\n" + "="*50)
    print("TESTING KSWIN WITH OBVIOUS STEP CHANGE")
    print("="*50)

    # Same data
    data = []
    for i in range(100):
        if i < 50:
            data.append(0.0)
        else:
            data.append(100.0)

    alphas = [0.1, 0.05, 0.01, 0.001]

    for alpha in alphas:
        print(f"\nTesting alpha = {alpha}")
        detector = drift.KSWIN(alpha=alpha, window_size=50, stat_size=20)

        detections = []
        for i, value in enumerate(data):
            detector.update(value)
            if detector.drift_detected:
                detections.append(i)
                print(f"  Change detected at step {i}")

        print(f"  Total detections: {len(detections)}")

        if detections:
            print(f"  SUCCESS: KSWIN can detect obvious changes")
            return True

    print("\nKSWIN COMPLETELY FAILED!")
    return False


def test_with_noise():
    """Test with slight noise to make it more realistic"""
    print("\n" + "="*50)
    print("TESTING WITH SLIGHT NOISE")
    print("="*50)

    np.random.seed(42)
    data = []
    for i in range(100):
        if i < 50:
            data.append(np.random.normal(0.0, 0.1))  # Mean 0, small noise
        else:
            data.append(np.random.normal(10.0, 0.1))  # Mean 10, small noise

    print(f"Data: first 50 ~ N(0, 0.1), next 50 ~ N(10, 0.1)")
    print(f"Sample values: {data[:5]} -> {data[50:55]}")

    # Test ADWIN with very sensitive settings
    print(f"\nTesting ADWIN with very sensitive delta=0.00001")
    detector = drift.ADWIN(delta=0.00001)

    detections = []
    for i, value in enumerate(data):
        detector.update(value)
        if detector.drift_detected:
            detections.append(i)
            print(f"  Change detected at step {i}")

    print(f"Total detections: {len(detections)}")

    if detections:
        print("SUCCESS: River library works with noisy data")
        return True
    else:
        print("FAILED: Even with noise, no detection")
        return False


def main():
    print("MINIMAL RIVER LIBRARY TEST")
    print("Testing if River's drift detection works at all")

    # Test 1: Most obvious change possible
    adwin_works = test_adwin_basic()

    # Test 2: Page-Hinkley
    ph_works = test_page_hinkley_basic()

    # Test 3: KSWIN
    ks_works = test_kswin_basic()

    # Test 4: With realistic noise
    noise_works = test_with_noise()

    print("\n" + "="*60)
    print("FINAL DIAGNOSIS")
    print("="*60)

    if any([adwin_works, ph_works, ks_works, noise_works]):
        print("✓ River library CAN detect changes")
        print("✓ Problem is likely in our implementation or data preprocessing")
        print("\nNext steps:")
        print("1. Check feature preprocessing")
        print("2. Verify parameter ranges")
        print("3. Check if PANN features have sufficient variation")
    else:
        print("✗ River library CANNOT detect even obvious changes")
        print("✗ Possible issues:")
        print("1. River installation problem")
        print("2. Incompatible River version")
        print("3. Fundamental misunderstanding of API")

        print("\nTroubleshooting:")
        print("pip install --upgrade river")
        print("or try different drift detection library")


if __name__ == "__main__":
    main()