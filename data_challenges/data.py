from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import SimpleNamespace
from typing import List, Union

import dask.dataframe as dd
import pandas as pd
from pandas import DataFrame


@dataclass
class PATHs:
    REPO_PATH = Path(__file__).parent.parent
    DATA_PATH = REPO_PATH.joinpath("data")

    TRAIN_A_PATH = DATA_PATH.joinpath("training/")
    TRAIN_B_PATH = DATA_PATH.joinpath("training_setB/")

    # a_files
    train_a_files = list(TRAIN_A_PATH.glob("*.psv"))
    # b_files
    train_b_files = list(TRAIN_B_PATH.glob("*.psv"))

    # a_merged
    train_a_merge_file = DATA_PATH.joinpath("train_a_merged.csv")
    # b_merged
    train_b_merge_file = DATA_PATH.joinpath("train_b_merged.csv")

    # all_merged
    train_all_merge_file = train_a_merge_file.with_name("train_all_merged.csv")

    results = REPO_PATH.joinpath("results")
    results_pp = results.joinpath("PandasProfileReport")
    results_dimred = results.joinpath("dim_redus")
    results_imp = results.joinpath("imputations")
    results_vis = results.joinpath("vis")


def prep_csv_merge(
    inputs: List[Path],  # List[PathLike],
    dst: Union[PathLike, str],
    # replace = True,  # {"|": ";", "NaN": ""}
) -> None:
    """
    Merges all the individual patient .psv files and removes NaNs.
    Replacing the NaNs with an empty string also decreases the filesize.

    Args:
        inputs: list of PhysioNet 2019 files.
        dst: output file.

    Examples:
        >>> prep_csv_merge(inputs = PATHs.train_a_files + PATHs.train_b_files, dst = "train_all_merged.csv")
        >>> prep_csv_merge(inputs = PATHs.train_a_files, dst = "data/train_a_merged.csv")
        >>> prep_csv_merge(inputs = PATHs.train_b_files, dst = "data/train_b_merged.csv")

        # Verify a/b_merged vs all_merged
        >>> df = pd.read_csv(PATHs.train_all_merge_file, delimiter = "|")
        >>> dfa = pd.read_csv(PATHs.train_a_merge_file, delimiter = "|")
        >>> pd.testing.assert_frame_equal(df[df.PatientID < 100_000], dfa)
        >>> dfb = pd.read_csv(PATHs.train_b_merge_file, delimiter = "|")
        >>> pd.testing.assert_frame_equal(df[df.PatientID >= 100_000], dfb)
    """
    inputs = [Path(x) for x in inputs]
    with open(dst, "w") as f:
        # write the columns names with prepended column for PatientID
        f.write("PatientID|" + inputs[0].read_text().splitlines()[0] + "\n")
        # write the content of all provided files (skipping the first line)
        [
            f.writelines([
                f'{p.stem[1:]}|{l}\n'  # prepend PatientID
                for l in p.read_text().replace("NaN", "").splitlines()[1:]
            ]) for p in inputs
        ]


def load_challenge_data(
    file,
    sep: str = None,
    use_dask: bool = False,
    preprocess: bool = True,
    **kwargs
) -> DataFrame:
    """
    Load the PhysioNet 2019 challenge data into a DataFrame.

    Args:
        file: path to the merged .csv file.
        sep:
        use_dask:
        kwargs: see pandas (https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

    Returns:
        DataFrame: [description]

    Examples:
        >>> dfa = load_challenge_data(PATHs.train_a_merge_file, sep = "|")
        >>> dfb = load_challenge_data(PATHs.train_b_merge_file, sep = "|")
        >>> df = load_challenge_data(PATHs.train_all_merge_file, sep = ";")
    """
    # try to guess the separator from | or ;
    if sep is None:
        if Path(
            list(file)[0] if isinstance(file, Iterable) else file
        ).read_text().find("|") > -1:
            sep = "|"
        else:
            sep = ";"

    if use_dask:
        # df = xr.DataArray(PATHs.train_a_files[:5])
        # df = dd.read_csv('data*.csv', delimiter = sep)
        df = dd.read_csv(file, delimiter = sep)
        # df = df.compute()

        # dfa = pd.read_csv(PATHs.train_a_merge_file, delimiter = sep)
        # dfb = pd.read_csv(PATHs.train_b_merge_file, delimiter = sep)
        # df = pd.concat([dfa, dfb])
    else:
        df = pd.read_csv(
            file,
            delimiter = sep,
            dtype = {
                # "PatientID": 'category',  # https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
                # "Gender": 'category',
                # "Unit1": 'category',
                # "Unit2": 'category',
                # "SepsisLabel": 'category',

                # "Gender": pd.BooleanDtype(),
                # "Unit1": pd.BooleanDtype(),
                # "Unit2": pd.BooleanDtype(),
                # "SepsisLabel": pd.BooleanDtype(),
            },
            # skiprows = kwargs.get("skiprows"),
            # nrows = kwargs.get("nrows"),
            **kwargs
        )
    # df = df.convert_dtypes()  # infers dtypes
    # print(df.dtypes)
    # df.agg([lambda s: s.round() == s]).agg("all")  # check if all values are integers


    if preprocess:
        df = rearrange_df(df)
        df = derive_data(df)

    return df


def load_data_set(data_set = "all", **kwargs) -> DataFrame:
    """
    Loads Dataset A and preprocesses it.

    Args:
        data_set: choose a dataset "a" or "b"
        kwargs: see load_challenge_data()

    Returns:
        [type]: [description]

    Examples:
        >>> dfa = load_data_set("A")
        >>> dfb = load_data_set("B")
        # check validity
        >>> pd.testing.assert_frame_equal(load_data_set("A"), load_challenge_data(PATHs.train_a_merge_file))
    """
    # find the line where dataset B starts (B has PatientIDs > 100_000)
    B_start_ix = [x[0] for x in PATHs.train_all_merge_file.read_text().splitlines()].index("1")

    if data_set.upper() == "A":
        kwargs["nrows"] = B_start_ix - 1
    elif data_set.upper() == "B":
        kwargs["skiprows"] = range(1, B_start_ix)
    # else:
    #     print("Loading datasets A & B")
    #     # raise ValueError("data_set must be either 'A' or 'B'")

    df = load_challenge_data(
        PATHs.train_all_merge_file,
        sep = ";",
        **kwargs
        )

    return df


def rearrange_df(df: DataFrame) -> DataFrame:
    """
    Prepares the DataFrame by adding Multiindex for patients.

    Args:
        df:

    Returns:
        DataFrame: [description]

    Examples:
        >>> df = rearrange_df(df)
    """

    # df.pivot_table(index = ["PatientID"])
    # df.groupby(by = "PatientID")

    df = df.reset_index().set_index(keys = ["PatientID", "index"])  # set multiindex
    labels = df.groupby(level = 0).cumcount()  # labels for level 1 index
    # df.set_index([df.index.get_level_values(0), labels]).loc[slice(2, None)][:50]
    df = df.droplevel(1).set_index([labels], append = True)  # reset level 1 index for each person
    #.loc[slice(2, None)][:50]
    # df.reset_index().describe()
    return df


def derive_data(df: DataFrame) -> DataFrame:
    """
    Adds additional features derived from the data.
    Adds these features:
        - Sepsis: does the patient ever develop sepsis?
        - Hours: duration of all measurements per patient
            (not to be confused with last ICULOS entry of each patient!)
        -

    Args:
        df:

    Returns:
        DataFrame: [description]

    Examples:
        >>> df = derive_data(df)
    """
    # Patient developed sepsis
    sepsis = df.groupby(level = 0).SepsisLabel.max()  #.astype(bool)
    sepsis.index = pd.Index((c, ) for c in sepsis.index)  # add a level to the index
    df["Sepsis"] = sepsis

    # anzahl der rows per patient <=> hours of stay
    rows_per_patient = df.reset_index(level = 1).level_1.groupby(level = 0).count()
    # print(f'{rows_per_patient.mean() = :.2f}')
    rows_per_patient.index = pd.Index((c, ) for c in rows_per_patient.index)
    df["Hours"] = rows_per_patient

    # df.iloc[:, -5:]
    # df.loc[df.loc[df.SepsisLabel.values == 1].index.get_level_values(0).unique()].iloc[:, -5:]
    return df


def sample_patients(df: Union[str, DataFrame, None] = None,
                    frac: float = 0.1) -> DataFrame:
    """
    Samples whole patients

    Args:
        df:
        frac:

    Returns:
        [type]: [description]

    Examples:
        >>> dfsample = sample_patients(dfa, frac = 0.1)
    """
    if isinstance(df, str):
        df = load_data_set(df)
    if df is None:
        df = load_data_set()

    patientIDs = df.index.unique("PatientID").to_frame().sample(frac = frac).PatientID
    patients_sample = df.loc[patientIDs]
    patients_sample = patients_sample.sort_index(level = 0)
    return patients_sample


# def drop_data(df: DataFrame) -> DataFrame:
# def impute_data(df: DataFrame) -> DataFrame:


if __name__ == "__main__":
    prep_csv_merge(inputs = PATHs.train_a_files + PATHs.train_b_files, dst = "train_all_merged.csv")
    prep_csv_merge(inputs = PATHs.train_a_files, dst = "data/train_a_merged.csv")
    prep_csv_merge(inputs = PATHs.train_b_files, dst = "data/train_b_merged.csv")