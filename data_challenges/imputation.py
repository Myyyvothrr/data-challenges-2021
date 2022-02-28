import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Tuple

from data_challenges.data import load_data_set
# from data import load_data_set
from data_challenges.dim_redu import dim_red


def impute_prepare(df: DataFrame) -> Tuple[DataFrame, list, list]:
    """
    Prepares dataframe for imputation:
    * deep copy
    * detect numerical and categorical features
    * remove features from list that are completely NA
    """
    # work on copy
    dfi = df.copy(deep = True)

    # list of all categorical features
    categorical_features = [
            "Gender",
            "Unit1",
            "Unit2",
            "SepsisLabel",
            "Sepsis",
            "PatientID",
    ]

    # list of only NA features
    na_cols = dfi.columns[dfi.isna().all()].tolist()

    # all numerical columns to imputate
    numerical_features_to_impute = list(set(dfi.columns) - set(categorical_features) - set(na_cols))

    # list of all categorical features to impute
    categorical_features_to_impute = [
            "Gender",
            "Unit1",
            "Unit2",
            "SepsisLabel"
    ]
    categorical_features_to_impute = list(set(categorical_features_to_impute) - set(na_cols))

    return dfi, categorical_features_to_impute, numerical_features_to_impute


def imputate_simple(df: DataFrame,
                    strategy: str) -> DataFrame:
    """
    Simple imputation based on values of the specific feature ("univariate") without grouping
    """

    dfi, categorical_features, numerical_features = impute_prepare(df)

    # impute numerical values based on selected strategy
    imputer_num = SimpleImputer(missing_values = np.nan, strategy = strategy)
    dfi[numerical_features] = imputer_num.fit_transform(dfi[numerical_features])

    # impute categorical features, always use most freq strategy
    imputer_cat = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
    dfi[categorical_features] = imputer_cat.fit_transform(dfi[categorical_features])

    return dfi


def impute_knn(df: DataFrame,
               neighbors: int,
               weights: str) -> DataFrame:
    """
    Imputation based on KNN without grouping
    """

    dfi, categorical_features, numerical_features = impute_prepare(df)

    # impute numerical values based on selected strategy
    imputer_num = KNNImputer(missing_values = np.nan, n_neighbors = neighbors, weights = weights)
    dfi[numerical_features] = imputer_num.fit_transform(dfi[numerical_features])

    # impute categorical features, always use most freq strategy
    imputer_cat = KNNImputer(missing_values = np.nan, n_neighbors = neighbors, weights = weights)
    dfi[categorical_features] = imputer_cat.fit_transform(dfi[categorical_features])

    return dfi


def _impute_simple_grouped_apply(group,
                                 strategy) -> DataFrame:
    """
    [summary]

    Args:
        group:
        strategy:

    Returns:
        DataFrame: [description]
    """
    dfi, categorical_features, numerical_features = impute_prepare(group)

    # impute numerical values based on selected strategy
    imputer_num = SimpleImputer(missing_values = np.nan, strategy = strategy)
    dfi[numerical_features] = imputer_num.fit_transform(dfi[numerical_features])

    # impute categorical features, always use most freq strategy
    imputer_cat = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
    dfi[categorical_features] = imputer_cat.fit_transform(dfi[categorical_features])

    return dfi


def impute_simple_grouped(df: DataFrame,
                          strategy: str,
                          group_name: str,
                          mask_ranges = None) -> DataFrame:
    """Simple imputation based on values of the specific feature ("univariate") with grouping"""

    # work on copy
    dfi = df.copy(deep = True)

    # ranges handle separately
    if mask_ranges is not None:
        for mask_range in mask_ranges:
            mask = (dfi[group_name] >= mask_range[0]) & (dfi[group_name] < mask_range[1])
            dfi.loc[mask] = _impute_simple_grouped_apply(dfi.loc[mask], strategy)

    else:
        # group
        if group_name in dfi.index.names:
            # already in multi index, ignore ranges
            dfi = dfi.groupby(level = group_name)
        else:
            dfi = dfi.groupby(by = group_name)

        # impute
        dfi = dfi.apply(lambda group: _impute_simple_grouped_apply(group, strategy))

    return dfi


def main_imputation():
    dfa = load_data_set("A", preprocess = True)

    # tsne cant handle NA?
    # dim_red(dfa, "tsne", "", 0.5, True)
    dim_red(dfa, "pacmap", "", 0.5)

    age_groups = [
            [0, 19],
            [19, 31],
            [31, 51],
            [51, 71],
            [71, 81],
            [81, 200]
    ]

    stay_ranges = [
            [0, 25],
            [25, 49],
            [49, 169],
            [169, 10000]
    ]

    # simple imputation:
    # univariate, whole dataset without subgroups
    # dfa_i_simple_mean = imputate_simple(dfa, "mean")
    # dim_red(dfa_i_simple_mean, "tsne", "i_simple_mean", 0.5)
    # dim_red(dfa_i_simple_mean, "tsne", "i_simple_mean", 1)
    # dim_red(dfa_i_simple_mean, "pacmap", "i_simple_mean", 0.5)
    # dim_red(dfa_i_simple_mean, "pacmap", "i_simple_mean", 1.0)
    # saving is ~500mb
    # dfa_i_simple_mean.to_csv("data/imputations/train_a_imputation_simple_mean.csv", sep = '|')
    # dfa_i_simple_median = imputate_simple(dfa, "median")
    # dfa_i_simple_most_frequent = imputate_simple(dfa, "most_frequent")
    # dim_red(dfa_i_simple_most_frequent, "pacmap", "i_simple_most_frequent", 0.5)
    # dim_red(dfa_i_simple_most_frequent, "pacmap", "i_simple_most_frequent", 1.0)

    # simple imputation:
    # univariate, whole dataset with subgroups
    dfa_i_simple_group_patient_mean = impute_simple_grouped(dfa, "mean", "PatientID")
    # dfa_i_simple_group_patient_mean.to_csv("data/imputations/train_a_imputation_simple_group_patient_mean.csv", sep = '|')
    dim_red(dfa_i_simple_group_patient_mean, "pacmap", "i_simple_group_patient_mean", 0.5)
    # dim_red(dfa_i_simple_group_patient_mean, "pacmap", "i_simple_group_patient_mean", 1.0)
    # dfa_i_simple_group_patient_median = impute_simple_grouped(dfa, "median", "PatientID")
    # dfa_i_simple_group_patient_most_frequent = impute_simple_grouped(dfa, "most_frequent", "PatientID")

    dfa_i_simple_group_age_mean = impute_simple_grouped(dfa, "mean", "Age")
    # dfa_i_simple_group_age_mean.to_csv("data/imputations/train_a_imputation_simple_group_age_mean.csv", sep = '|')
    dim_red(dfa_i_simple_group_age_mean, "pacmap", "i_simple_group_age_mean", 0.5)
    # dim_red(dfa_i_simple_group_age_mean, "pacmap", "i_simple_group_age_mean", 1.0)
    # dfa_i_simple_group_age_median = impute_simple_grouped(dfa, "median", "Age")
    # dfa_i_simple_group_age_most_frequent = impute_simple_grouped(dfa, "most_frequent", "Age")

    dfa_i_simple_group_gender_mean = impute_simple_grouped(dfa, "mean", "Gender")
    # dfa_i_simple_group_gender_mean.to_csv("data/imputations/train_a_imputation_simple_group_gender_mean.csv", sep = '|')
    dim_red(dfa_i_simple_group_gender_mean, "pacmap", "i_simple_group_gender_mean", 0.5)
    # dim_red(dfa_i_simple_group_gender_mean, "pacmap", "i_simple_group_gender_mean", 1.0)
    # dfa_i_simple_group_gender_median = impute_simple_grouped(dfa, "median", "Gender")
    # dfa_i_simple_group_gender_most_frequent = impute_simple_grouped(dfa, "most_frequent", "Gender")

    dfa_i_simple_group_age_ranges_mean = impute_simple_grouped(dfa, "mean", "Age", age_groups)
    # dfa_i_simple_group_age_ranges_mean.to_csv("data/imputations/train_a_imputation_simple_group_age_ranges_mean.csv", sep = '|')
    dim_red(dfa_i_simple_group_age_ranges_mean, "pacmap", "i_simple_group_age_ranges_mean", 0.5)
    # dim_red(dfa_i_simple_group_age_ranges_mean, "pacmap", "i_simple_group_age_ranges_mean", 1.0)

    # nearest neighbor imputation:
    # whole dataset without subgroups
    # TODO testen, läuft zu lange -> erstmal abgebrochen
    # dfa_i_knn_uniform_5 = impute_knn(dfa, 5, "uniform")
    # dfa_i_knn_uniform_5.to_csv("data/imputations/train_a_imputation_knn_uniform_5.csv", sep = '|')
    # dfa_i_knn_uniform_3 = impute_knn(dfa, 3, "uniform")
    # dfa_i_knn_uniform_3.to_csv("data/imputations/train_a_imputation_knn_uniform_3.csv", sep = '|')

    # dfa_i_knn_distance_5 = impute_knn(dfa, 5, "distance")
    # dfa_i_knn_distance_3 = impute_knn(dfa, 3, "distance")
    # dfa_i_knn_distance_3.to_csv("data/imputations/train_a_imputation_knn_distance_3.csv", sep = '|')

    print("done")


if __name__ == "__main__":
    main_imputation()
