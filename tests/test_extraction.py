import sys
import os

# Add project root and src directory to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.append(project_root)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.main import extract_final_answer, is_correct, extract_numeric_value


def test_extraction():
    all_passed = True

    # 1. Direct tests for extract_numeric_value()
    print("Running Direct extract_numeric_value Tests:")
    numeric_cases = [
        ("15", "15"),
        ("-15", "-15"),
        ("$24", "24"),
        ("2,125", "2125"),
        ("24.50%", "24.50"),
        ("24.5", "24.5"),
        ("", "N/A"),
        (None, "N/A"),
        ("abc", "N/A"),
    ]
    for val, expected in numeric_cases:
        res = extract_numeric_value(val)
        if res == expected:
            print(f"  [PASS] extract_numeric_value({repr(val)}) -> {res}")
        else:
            print(f"  [FAIL] extract_numeric_value({repr(val)}) -> {res} (Expected: {expected})")
            all_passed = False

    # 2. Extract final answer tests
    print("\nRunning extract_final_answer Tests:")
    final_answer_cases = [
        ("FINAL ANSWER: 15", "15"),
        ("FINAL ANSWER: -15", "-15"),
        ("FINAL ANSWER: $24", "24"),
        ("FINAL ANSWER: 2,125", "2125"),
        ("FINAL ANSWER: 24.5", "24.5"),
        ("FINAL ANSWER: 20%", "20"),
        ("Some steps\nAnother step\nFINAL ANSWER: 42", "42"),
        ("No explicit final answer but ending with 100", "100"),
        ("No numbers here at all", "N/A"),
        ("FINAL ANSWER:", "N/A"),
        ("FINAL ANSWER: abc", "abc"),
    ]
    for val, expected in final_answer_cases:
        res = extract_final_answer(val)
        if res == expected:
            print(f"  [PASS] extract_final_answer({repr(val)}) -> {res}")
        else:
            print(f"  [FAIL] extract_final_answer({repr(val)}) -> {res} (Expected: {expected})")
            all_passed = False

    # 3. Numeric equivalence and tolerance boundary tests
    print("\nRunning is_correct and Tolerance Tests:")
    correctness_cases = [
        ("15", "15", True),
        ("-15", "-15", True),
        ("2,125", "2125", True),
        ("$24", "24", True),
        ("24.5", "24.5", True),
        ("24", "24.0", True),
        ("-15", "-15.0", True),
        ("24.500", "24.505", True),
        ("24.500", "24.511", False),
        ("24.499", "24.490", True),
        ("N/A", "N/A", True),
        ("abc", "abc", True),
        ("abc", "def", False),
    ]
    for p, a, expected in correctness_cases:
        res = is_correct(p, a)
        if res == expected:
            print(f"  [PASS] is_correct({repr(p)}, {repr(a)}) -> {res}")
        else:
            print(f"  [FAIL] is_correct({repr(p)}, {repr(a)}) -> {res} (Expected: {expected})")
            all_passed = False

    if all_passed:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    test_extraction()