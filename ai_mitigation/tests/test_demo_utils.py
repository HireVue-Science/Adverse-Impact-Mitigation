import numpy as np
import pandas as pd

from ai_mitigation.demo_utils import convert_demo_dict_to_series, convert_series_to_demo_dict


def assert_demo_dict_equals(d1, d2):
    assert d1.keys() == d2.keys()
    for demo_name, arr1 in d1.items():
        arr2 = d2[demo_name]
        assert np.array_equal(arr1, arr2)


def test_convert_series_to_demo_dict():
    expected_demo_dict = {
        "Male": np.array([True, False, False]),
        "Female": np.array([False, True, False]),
    }

    demo = ["Male", "Female", ""]
    demo_dict = convert_series_to_demo_dict(demo)
    assert_demo_dict_equals(demo_dict, expected_demo_dict)

    demo = pd.Series(["Male", "Female", ""])
    demo_dict = convert_series_to_demo_dict(demo)
    assert_demo_dict_equals(demo_dict, expected_demo_dict)

    demo = pd.DataFrame(data={"demo_Gender": ["Male", "Female", ""]})
    demo_dict = convert_series_to_demo_dict(demo)
    assert_demo_dict_equals(demo_dict, expected_demo_dict)

    demo_dict = convert_series_to_demo_dict(expected_demo_dict)
    assert_demo_dict_equals(demo_dict, expected_demo_dict)

    assert not convert_series_to_demo_dict({})


def test_convert_demo_dict_to_series():
    demo = ["Male", "Female", ""]
    index = [10, 20, 30]
    s1 = pd.Series(demo)
    s2 = pd.Series(demo, index=index)

    demo_dict = {
        "Male": np.array([True, False, False]),
        "Female": np.array([False, True, False]),
    }

    s = convert_demo_dict_to_series(demo_dict)
    assert s.equals(s1)
    s = convert_demo_dict_to_series(demo_dict, index)
    assert s.equals(s2)
    s = convert_demo_dict_to_series(s1)
    assert s.equals(s1)
    s = convert_demo_dict_to_series(demo)
    assert s.equals(s1)
    s = convert_demo_dict_to_series(demo, index)
    assert s.equals(s2)
