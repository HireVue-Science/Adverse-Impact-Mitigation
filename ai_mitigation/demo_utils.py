import numpy as np
import pandas as pd


def convert_series_to_demo_dict(ser):
    """converts a pandas series/list/array to a demo dictionary.
    if a demo_dict is passed in, it returns it"""
    if isinstance(ser, pd.DataFrame):  # convert 1d DataFrame to Series
        assert ser.shape[1] == 1, "Dataframe has multiple demographic columns"
        ser = ser.iloc[:, 0]
    elif isinstance(ser, dict):
        demo_dict = ser
        if len(demo_dict) == 0:
            return demo_dict
        demo_object = demo_dict[next(iter(demo_dict.keys()))]
        assert not isinstance(
            demo_object, dict
        ), "Only pass in a single demographic"  # multiple dicts
        return demo_dict

    ser = pd.Series(ser)  # handle list, array, series

    demo_dict = {}

    ser = ser.fillna(value="")
    assert ser.map(type).eq(str).all(), f"column '{ser.name}' is not a valid demographic column"
    demo_groups = [demo_group for demo_group in np.unique(ser) if demo_group != ""]
    for demo_group in demo_groups:
        demo_dict[demo_group] = (ser == demo_group).values

    return demo_dict


def convert_df_to_demo_dicts(df, columns=None):
    """converts a DataFrame to a full demo_dicts (dictionary where each value is a single
    demo_dict). If a full demo_dicts dictionary is passed in, it returns it
    """
    if isinstance(df, dict):
        demo_dicts = df
        demo_object = demo_dicts[next(iter(demo_dicts.keys()))]
        assert isinstance(demo_object, dict), "pass in full demographics"  # single dict of masks
        return demo_dicts

    df = pd.DataFrame(df)  # convert Series
    if columns is None:
        columns = get_demo_columns(df)
    assert len(columns) > 0, (
        "No valid demographic columns found. Either "
        "specify your columns, or choose valid demographic column names"
    )

    demo_dicts = {}
    for col in columns:
        demo_dicts[col] = convert_series_to_demo_dict(df[col])
    return demo_dicts


def convert_demo_dict_to_series(demo_dict, index=None):
    """converts a single demo_dict to a demo_series.

    A demo_dict is a dictionary where each key is a demographic group
    (e.g. Male/Female/Nonbinary), and the corresponding values are boolean arrays
    denoting the roup membership

    index (optional): used to set the dataframe index

    A pandas series with string demographic groups is returned
    """

    if isinstance(demo_dict, pd.Series):
        return demo_dict

    if not isinstance(demo_dict, dict):  # list/array
        if index is None:
            N = len(demo_dict)
            index = np.arange(N)
        return pd.Series(demo_dict, index=index)

    if index is None:
        N = len(demo_dict[next(iter(demo_dict.keys()))])
        index = np.arange(N)
    s = pd.Series("", index=index)
    for demo_group, demo_mask in demo_dict.items():
        assert isinstance(demo_mask, np.ndarray)
        s.loc[demo_mask] = demo_group
    return s


def convert_demo_dicts_to_df(demo_dicts, index=None):
    """takes a full demo dict and returns a dataframe of demographics"""
    if isinstance(demo_dicts, pd.DataFrame):
        return demo_dicts

    demo_df = pd.DataFrame(index=index)
    for demo_category, demo_dict in demo_dicts.items():
        demo_df[demo_category] = convert_demo_dict_to_series(demo_dict, index)
    return demo_df


def get_demo_columns(df):
    """A demographic DataFrame might contain non-demographic columns (e.g. training target), so by
    convention, all demographic columns must start with "demo_", e.g. "demo_Gender".
    """
    return [col for col in df.columns if (isinstance(col, str) and col.startswith("demo_"))]


def prune_demo_dict(demo_dict, valid_options=None):
    """Removes small groups from demo dict. valid_options takes the form
    {"min_size": 20, "min_frac": 0.01}
    """

    assert isinstance(demo_dict, dict)
    if valid_options is None:
        valid_options = {}
    new_demo_dict = {}
    valid_groups = calc_valid_groups(demo_dict, **valid_options)
    for label, mask in demo_dict.items():
        if label in valid_groups:
            new_demo_dict[label] = mask
    return new_demo_dict


def prune_demo_dicts(demo_dicts, valid_options=None):
    """prunes each demo_dict,
    but also removes a demo_dict that only has one demo group

    valid_options takes the form:
    {"min_size": 20, "min_frac": 0.01}

    """
    if valid_options is None:
        valid_options = {}

    pruned_demo_dicts = {}
    for demo_category, demo_dict in demo_dicts.items():
        demo_dict = prune_demo_dict(demo_dict, valid_options)
        if len(demo_dict) > 1:
            pruned_demo_dicts[demo_category] = demo_dict
    return pruned_demo_dicts


def prune_demo(demo, valid_options=None):
    if isinstance(demo, dict):
        return prune_demo_dicts(demo, valid_options)
    # otherwise dataframe
    demo_dicts = convert_df_to_demo_dicts(demo)
    demo_dicts = prune_demo_dicts(demo_dicts, valid_options)
    demo_df = convert_demo_dicts_to_df(demo_dicts, demo.index)
    return demo_df


def calc_valid_groups(demo, min_size=20, min_frac=0.0, category_mask=None):
    """returns a list of valid group names in a demo_dict,
    given min_size and min_frac parameters"""

    demo_dict = convert_series_to_demo_dict(demo)

    if min_size == 0 and min_frac == 0:
        return list(demo_dict.keys())
    if category_mask is None:
        category_mask = np.logical_or.reduce(list(demo_dict.values()))
    N_total = np.sum(category_mask)

    valid_groups = []
    for demo_group, demo_mask in demo_dict.items():
        N_group = np.sum(demo_mask & category_mask)
        if N_group >= min_size and N_group / N_total >= min_frac:
            valid_groups.append(demo_group)

    return valid_groups


def _create_mask_pairs_from_demo(demo):
    """from a full demo dict (where
    full_demo_dict['category']['subcategory'] is a boolean array) or a
    demo dataframe, returns a list of (demo_mask, category_mask)
    pairs.

    The mask pairs are used for MPO models
    """
    demo_dicts = convert_df_to_demo_dicts(demo)
    mask_pairs = []
    for demo_dict in demo_dicts.values():
        category_mask = np.logical_or.reduce(list(demo_dict.values()))
        for mask in demo_dict.values():
            mask_pairs.append((np.asarray(mask, dtype="bool"), category_mask))
    return mask_pairs
