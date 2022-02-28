from pandas import DataFrame
from typing import Union, List
import pandas as pd
from data_challenges.data import load_data_set
from data_challenges.helper import is_ipython
from data_challenges.helper import pandas_set_auto_style


pandas_set_auto_style()


def find_low_quality_features(df: DataFrame = None,
                              dfa: DataFrame = None,
                              dfb: DataFrame = None,
                              frac: float = 0.5,
                              merged_set: bool = False,
                              use_set: str = "A|B",  # A B A|B All
                              show_missing: bool = False,
                              return_styles_df: bool = False,
                              return_df: bool = False,
                              verbose: bool = False):
    """
    Finds low quality features by counting the number of patients with at least one measurement.
    Dropping features with less than `frac` patients with at least one measurement in A or B.

    Args:
        df: A & B merged data set.
        dfa: A data set.
        dfb: B data set.
        frac: Fraction of patients with at least one measurement. Used to filter out low quality features.
        merged_set: Use merged data set instead of A & B when looking for `frac` patients with features.
        use_set: [A B A|B All]
        verbose: Visualization

    Returns:
        [type]: [description]

    Examples:
        >>> find_low_quality_features(frac = .5, verbose = True)
        >>> df = df.drop(find_low_quality_features(df, dfa, dfb, frac = .5), axis = 1)
    """
    # if use_set.lower() == "all":
    df = load_data_set("all") if df is None else df
    dfa = load_data_set("a") if dfa is None else dfa
    dfb = load_data_set("b") if dfb is None else dfb

    def rename_axes(df: DataFrame, missing = show_missing):
        return df.rename(
                columns = {
                        "A": "SetA",
                        "B": "SetB",
                        "All": "MergedSets",
                        "pct": "Threshold",
                        "count": "Missing_percentage" if missing else "Not_NaN_percentage",
                }
        )

    df_feat_counts = []
    for dfx, name in [(dfa, "A"), (dfb, "B"), (df, "All")]:
    # for dfx, name in [(dfa, "SetA"), (dfb, "SetB"), (df, "MergedSets")]:
        no_patients = len(dfx.index.unique("PatientID"))
        feature_count = (dfx.groupby(level = "PatientID").count() > 0).sum().to_frame(name)
        feature_pct = feature_count / no_patients

        df_feat_counts.append(
            pd.concat([feature_pct, feature_pct > frac], axis = 1, keys = ["count", "pct"]).swaplevel(0, 1, axis = 1)
            # pd.concat([feature_pct, feature_pct > frac], axis = 1, keys = ["Not_NaN_percentage", "Threshold"]).swaplevel(0, 1, axis = 1)
        )
        # pd.concat([feature_count, feature_count / no_patients > .5], axis = 1, ignore_index = True).style.bar()

    # pd.concat(df_feat_counts, axis = 1).T.style.use(df_auto_style)
    df_feat_counts = pd.concat(df_feat_counts, axis = 1)
    if show_missing:
        df_feat_counts = df_feat_counts.apply(lambda x: x if x.dtype == bool else (1 - x))
    if return_df:
        return rename_axes(df_feat_counts)
    if verbose or return_styles_df:
        try:
            # print(f'{__file__}: {hasattr(DataFrame, "style_auto") = }')
            df_styled = rename_axes(df_feat_counts).style_auto(precision = 3)
            if return_styles_df:
                return df_styled
            else:
                display(df_styled)
        except Exception:
            if is_ipython():
                display(df_feat_counts)
            else:
                print(df_feat_counts)

    # select all features that have less than 50% of the patients with at least one measurement
    if not merged_set:
        if use_set.lower() == "a":
            low_quality_features = df_feat_counts.A.index[df_feat_counts.A.pct <= frac]
        elif use_set.lower() == "b":
            low_quality_features = df_feat_counts.B.index[(df_feat_counts.B.pct <= frac)]
        elif use_set.lower() == "a|b":
            # in A or B
            low_quality_features = df_feat_counts.A.index[(df_feat_counts.A.pct <= frac) | (df_feat_counts.B.pct <= frac)]
    else:
        # in All
        low_quality_features = df_feat_counts.A.index[(df_feat_counts.All.pct <= frac)]

    return low_quality_features


def drop_low_quality_features(dfs: List[DataFrame],
                              df: DataFrame = None,
                              dfa: DataFrame = None,
                              dfb: DataFrame = None,
                              inplace = True,
                              **kwargs) -> Union[None, List[DataFrame]]:
    """
    Drop low quality features.

    Args:
        dfs: DataFrames where to drop low quality features.
        df: merged data set.
        dfa: A data set.
        dfb: B data set.

    Returns:
        List[DataFrame]: [description]

    Examples:
        >>> df, dfa, dfb = load_data_set("all"), load_data_set("a"), load_data_set("b")
        >>> drop_low_quality_features([df, dfa, dfb], df, dfa, dfb, frac = .5)
        >>> df, dfa, dfb = drop_low_quality_features([df, dfa, dfb], df, dfa, dfb, frac = .5, inplace = False)
    """
    low_quality_feats = find_low_quality_features(df, dfa, dfb, **kwargs)
    if inplace:
        for dfx in dfs:
            dfx.drop(low_quality_feats, axis = 1, inplace = True)
    else:
        dfs = [dfx.drop(low_quality_feats, axis = 1) for dfx in dfs]
        return dfs

# df, dfa, dfb = load_data_set("all"), load_data_set("a"), load_data_set("b")
# drop_low_quality_features([df, dfa, dfb, dfsample], df, dfa, dfb, frac = .5)


def drop_sparse_data_patients():
    # TODO
    raise NotImplementedError
