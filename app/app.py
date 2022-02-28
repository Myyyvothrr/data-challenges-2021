import json
from os import times_result
from tkinter.ttk import LabeledScale
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1
import pysubgroup as ps
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from pyts.classification import KNeighborsClassifier, TimeSeriesForest
from pyts.preprocessing import InterpolationImputer, StandardScaler

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from data_challenges.imputation import imputate_simple, impute_simple_grouped
from data_challenges.data import PATHs, load_data_set
from data_challenges.clean import find_low_quality_features
from data_challenges.helper import pandas_set_auto_style
from data_challenges.viz_dim_redu import bokeh_change_color_by_feat
from data_challenges.dim_redu import dim_red
from data_challenges.clustering import cluster, cluster_subspace
pandas_set_auto_style()
# print(f'{__file__}: {hasattr(DataFrame, "style_auto") = }')


@st.cache
def load_dataset(dataset_path: str):
    df = pd.read_csv(dataset_path,
                     sep = ";",
                     dtype = {
                             "Gender":      str,
                             "Unit1":       str,
                             "Unit2":       str,
                             "SepsisLabel": str,
                             "PatientID":   str
                     }
                     )

    return df


@st.cache
def load_data_sets():
    return load_data_set(), load_data_set("A"), load_data_set("B")


@st.cache
def find_subgroups(target_label: str,
                   target_value,
                   results: int,
                   depth: int,
                   sample_frac: float = .001,
                   ):
    target = ps.BinaryTarget(target_label, target_value)
    searchspace = ps.create_selectors(df.sample(frac = sample_frac, random_state = 1), ignore = [target_label, 'PatientID'])
    task = ps.SubgroupDiscoveryTask(df.sample(frac = sample_frac, random_state = 1), target, searchspace,
                                    result_set_size = results,
                                    depth = depth,
                                    qf = ps.WRAccQF()
                                    )
    result = ps.BeamSearch().execute(task)
    return result.to_dataframe()


@st.cache
def get_time_series(df, target_label, feature):
    time_index_max = 0
    time_series_temp = []
    labels = []
    for name, group in df.groupby(by=["PatientID"]):
        #patients.append(name)

        time_index_max = np.max([time_index_max, group["ICULOS"].max()])

        # use last value for target label
        labels.append(group[target_label].values[-1])

        features = {
            f["ICULOS"]: f[feature]
            for f
            in json.loads(
                group[["ICULOS", feature]].to_json(orient="records")
            )
        }
        time_series_temp.append(features)

    # reformat features to produce same length vectors
    time_series = []
    for ts in time_series_temp:
        tslist = []
        for i in range(1, time_index_max+1):
            app = False
            if i in ts:
                if ts[i] is not None:
                    tslist.append(ts[i])
                    app = True
            if not app:
                tslist.append(np.nan)

        time_series.append(tslist)

    del time_series_temp

    time_series = np.array(time_series)
    labels = np.array(labels).ravel()

    time_series = np.nan_to_num(time_series, nan=0)
    labels = np.nan_to_num(labels, nan=0)

    return time_series, labels


# set app title
st.set_page_config(page_title = "Data Challenges Project 2021", layout = "wide")
st.title("Data Challenges Project 2021")
st.write("Team DC1: Daniel Baumartz, Hendrik Edelmann, Kai Wo Yau")

# provide task selection to not load all tasks directly
all_tasks = [
    "Task 1: Dataset Exploration",
    "Task 2: Dimensionality Reduction and Imputation",
    "Task 3: Clustering",
    "Task 4: Subspace Clustering",
    "Task 5: Imbalanced Learning",
    "Task 6: Time Series Classification"
]
main_task_selection = st.selectbox("Choose task", all_tasks)

st.subheader(main_task_selection)

# Task 1
if main_task_selection == all_tasks[0]:
    # dataset
    st.header("Dataset")
    st.write("*Early Prediction of Sepsis from Clinical Data -- the PhysioNet Computing in Cardiology Challenge 2019*",
            "https://physionet.org/content/challenge-2019/1.0.0/")

    df = load_dataset(PATHs.DATA_PATH.joinpath("train_all_merged.csv"))
    dfm, dfa, dfb = load_data_sets()
    # dfsample = sample_patients(dfa, frac = 0.1)

    # basic stats
    samples_count, features_count = df.shape

    # numerical features
    description_numeric = df.describe()
    numerical_features_count = description_numeric.shape[1]

    # categorical features
    description_categorical = df.describe(include = ['object'])
    categorical_features_count = description_categorical.shape[1]
    patients_count = description_categorical.loc["unique"]["PatientID"]

    # display stats
    col1_1, col1_2, col1_3, col1_4 = st.columns([1.5, 1, 1, 1])
    with col1_1:
        st.metric("Samples", samples_count)
    with col1_2:
        st.metric("Features", features_count)
    with col1_3:
        st.metric("Numerical Features", numerical_features_count)
    with col1_4:
        st.metric("Categorical features", categorical_features_count)

    # more info about gender
    df_males = df.loc[df['Gender'] == '1']
    male_patients_count = len(df_males["PatientID"].unique())

    # more info about sepsis
    df_sepsis = df.loc[df['SepsisLabel'] == '1']
    sepsis_patients_count = len(df_sepsis["PatientID"].unique())

    # gender and sepsis
    df_males_with_sepsis = df.loc[(df['Gender'] == '1') & (df['SepsisLabel'] == '1')]
    male_sepsis_patients_count = len(df_males_with_sepsis["PatientID"].unique())

    # display
    col2_1, col2_2, col2_3, col2_4 = st.columns([1.5, 1, 1, 1])
    with col2_1:
        st.metric("Patients", patients_count)
    with col2_2:
        st.metric("Patients with sepsis", sepsis_patients_count)
    with col2_3:
        st.metric("Male patients", male_patients_count)
    with col2_4:
        st.metric("Male patients w/ sepsis", male_sepsis_patients_count)

    # more info
    st.write("The dataset is taken from two hospitals and contains hourly data from", patients_count,
            "patients, that is *Demographics*, *vital signs* and *laboratory values*. There are", samples_count,
            "samples (observations, lines) and", features_count, "features, that are split in", numerical_features_count,
            "numerical features, like the patients temperature, and", categorical_features_count,
            "categorical features like their gender.")

    # sepsis / gender
    sepsis_percent = sepsis_patients_count / patients_count
    male_sepsis_percent = male_sepsis_patients_count / sepsis_patients_count
    male_percent = male_patients_count / patients_count
    st.write("Only", np.around(sepsis_percent * 100, 2), "% of patients, or", sepsis_patients_count,
            ", in the dataset have developed sepsis,", male_sepsis_patients_count, "(",
            np.around(male_sepsis_percent * 100, 2), "%) of which are male, with", male_patients_count,
            "male patients total (", np.around(male_percent * 100, 2), ").")

    # show first lines, full dataset is much too large
    with st.expander("Show first lines of dataset"):
        st.dataframe(df.head(n = 100))

    with st.expander("📜 Pandas Profiling Reports"):
        st.write("We used [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling) on the dataset to get a"
                "quick overview of the data.",
                "Shown here are the reports for the datasets A and B separately.",
                "For both datasets we calculated the mean for each feature for all patients.",
                "\n\n",
                "For reports on differently prepared Datasets and fullscreen view have a look at the repository",
                "[Pandas Profiling Reports @GitHub Pages](https://myyyvothrr.github.io/data-challenges-2021/results/PandasProfileReport/)")
        col_pp_1, col_pp_2 = st.columns(2)
        with col_pp_1:
            # st.components.v1.iframe(list(PATHs.results_pp.glob("*.html"))[0].read_text())
            st.components.v1.html(list(PATHs.results_pp.glob("*agg-mean-A.html"))[0].read_bytes(),
                                scrolling = True, height = 600)
        with col_pp_2:
            st.components.v1.html(list(PATHs.results_pp.glob("*agg-mean-B.html"))[0].read_bytes(),
                                scrolling = True, height = 600)

    with st.expander("Show numerical features"):
        st.dataframe(description_numeric)

    # TODO: not working
    # with st.expander("Show categorical features"):
    #     st.dataframe(description_categorical)

    # plot age of patients
    # col3_1, col3_2 = st.columns(2)
    col3_1, col3_2 = st.columns(2)
    with col3_1:
        st.write("Histogram of patients ages for all genders (", 0, "= female,", 1, "= male):")
        patients_ages = df.groupby(by = ['PatientID'])[['Age', 'Gender']].first()
        fig, ax = plt.subplots()
        sns.histplot(patients_ages, x = 'Age', hue = 'Gender', kde = True, bins = 20)
        st.pyplot(fig)

    # info abount amount of NA for features
    nans = df.isna().sum().rename("NA amount")
    with col3_2:
        st.write("Amount of NA values")
        fig, ax = plt.subplots()
        sns.histplot(nans, kde = True, bins = 20)
        plt.ticklabel_format(style = 'plain', axis = 'x')
        st.pyplot(fig)

    with st.expander("Show amount of NA values per feature"):
        # st.table(nans.to_frame().style.bar())

        frac = st.slider("Threshold",
                        0.0, 1.0, value = 0.5,
                        help = "Threshold for max amount of NA values per feature.",
                        )


        @st.cache(allow_output_mutation = True)
        def get_df_na_vals(show_missing = True):
            dfr: pd.DataFrame = find_low_quality_features(
                    dfm, dfa, dfb, frac = 0.5, return_df = True, show_missing = show_missing
            )
            return dfr

        df_na_vals = get_df_na_vals()


        def calc_missing_na(dfr, fract = 0.5):
            for s in dfr.columns.unique(level = 0):
                dfr[(s, "Threshold")] = dfr.loc[:, (s, dfr.columns[0][1])] <= fract
            return dfr


        st.dataframe(
                # df_na_vals.style_auto(precision = 3),
                calc_missing_na(df_na_vals, frac).style_auto(precision = 3),
                height = 2000
        )

        fig = plt.figure(figsize = (15, 5))
        ax = fig.gca()
        sns.barplot(
                x = "level_0",
                y = "Missing_percentage",
                hue = "level_1",
                # data = x,
                data = calc_missing_na(df_na_vals, frac).stack(level = 0).reset_index(),
                ax = ax,
        )
        plt.axhline(frac, color = "m", linestyle = "-")
        ax.tick_params(axis = "x", rotation = 45)
        # plt.gcf().set_size_inches(15, 4)
        st.pyplot(fig)
        # TODO: better show missing values instead of "non missing"?
        # TODO: add missing matrix (from pandas profiling)

# Task 2
elif main_task_selection == all_tasks[1]:
    # Dimensionality reduction
    st.subheader("Data imputation")

    @st.cache
    def task2_load_dataset(dataset_name):
        # load data set
        df, dfa, dfb = load_data_sets()
        if dataset_name == "A":
            df = dfa
        elif dataset_name == "B":
            df = dfb
        return df

    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.selectbox("Dataset", ["A", "B", "All"])

    df = task2_load_dataset(dataset_name)

    age_groups = [
            [0, 19],
            [19, 31],
            [31, 51],
            [51, 71],
            [71, 81],
            [81, 200]
    ]

    @st.cache
    def task2_impute(df, imputation_method):
        if imputation_method == "simple_mean":
            return imputate_simple(df, "mean")
        elif imputation_method == "simple_median":
            return imputate_simple(df, "median")
        elif imputation_method == "simple_most_frequent":
            return imputate_simple(df, "most_frequent")
        elif imputation_method == "grouped_patient_mean":
            return impute_simple_grouped(df, "mean", "PatientID")
        elif imputation_method == "grouped_age_mean":
            return impute_simple_grouped(df, "mean", "Age")
        elif imputation_method == "grouped_age_ranges_mean":
            return impute_simple_grouped(df, "mean", "Age", age_groups)
        elif imputation_method == "grouped_gender_mean":
            return impute_simple_grouped(df, "mean", "Gender")
        return df
    
    with col2:
        imputation_method = st.selectbox("Imputation method", ["None", "simple_mean", "simple_median", "simple_most_frequent", "grouped_patient_mean", "grouped_age_mean", "grouped_age_ranges_mean", "grouped_gender_mean"])

    df_imputed = task2_impute(df, imputation_method)

    with st.expander("Comparison with imputed dataset"):
        col1, col2 = st.columns(2)
        n_rows = 50
        with col1:
            st.write("Samples:", df.shape[0], ", features: ", df.shape[1])
            st.write(df.describe())
            st.write("Raw data of first", n_rows, "rows:")
            st.write(df.head(n=n_rows))
        with col2:
            st.write("Samples:", df_imputed.shape[0], ", features: ", df_imputed.shape[1])
            st.write(df_imputed.describe())
            st.write("Imputed data of first", n_rows, "rows:")
            st.write(df_imputed.head(n=n_rows))

    @st.cache
    def task2_dimred(df, dimred_method, sample_size):
        return dim_red(df, dimred_method, sample_size=sample_size, return_for_app=True)

    st.subheader("Dimensionality reduction")
    col1, col2, col3 = st.columns(3)

    with col1:
        sample_size = st.slider("Sample size", 0.01, 1.0, 0.1)

    with col2:
        dimred_method = st.selectbox("Dimensionality reduction method", ["--Choose--", "pacmap", "tsne", "umap"])
    
    with col3:
        selected_feature = st.selectbox("Show feature", df_imputed.columns.tolist(), index=df_imputed.shape[1]-2)

    if dimred_method == "--Choose--":
        st.write("Please choose a dimensionality reduction method.")
    else:
        st.write("Results of ", dimred_method)
        df_temp, dimredu_label = task2_dimred(df_imputed, dimred_method, sample_size)
        fig, ax = plt.subplots()
        sns.scatterplot(
                data = df_temp,
                x = dimredu_label[0],  # "dim-1",
                y = dimredu_label[1],  # "dim-2",
                hue = selected_feature,
                #palette = sns.color_palette('bright'),
                alpha = 0.5
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)


    st.subheader("Precalculated interactive visualization")
    with st.expander("show"):
        # bokeh_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP_imputed-iter-60-mean-asc-all-sample0.01.parquet"))
        # https://docs.bokeh.org/en/latest/docs/user_guide/embed.html#bokeh-applications
        # st.components.v1.html("""<embed src="http://localhost:5006/" style="width: 100%; height: 900px">""", scrolling = True, height = 1000)
        st.components.v1.html(PATHs.results_dimred.joinpath("viz_dimred_interactive.html").read_bytes(), scrolling = True, height = 1000)


# Task 3
elif main_task_selection == all_tasks[2]:
    st.subheader("Data imputation")

    @st.cache
    def task3_load_dataset(dataset_name):
        # load data set
        df, dfa, dfb = load_data_sets()
        if dataset_name == "A":
            df = dfa
        elif dataset_name == "B":
            df = dfb
        return df

    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.selectbox("Dataset", ["A", "B", "All"])

    df = task3_load_dataset(dataset_name)

    @st.cache
    def task3_impute(df, imputation_method):
        if imputation_method == "simple_mean":
            return imputate_simple(df, "mean")
        elif imputation_method == "simple_median":
            return imputate_simple(df, "median")
        elif imputation_method == "simple_most_frequent":
            return imputate_simple(df, "most_frequent")
        elif imputation_method == "grouped_patient_mean":
            return impute_simple_grouped(df, "mean", "PatientID")
        elif imputation_method == "grouped_age_mean":
            return impute_simple_grouped(df, "mean", "Age")
        elif imputation_method == "grouped_age_ranges_mean":
            return impute_simple_grouped(df, "mean", "Age", age_groups)
        elif imputation_method == "grouped_gender_mean":
            return impute_simple_grouped(df, "mean", "Gender")
        return df
    
    with col2:
        imputation_method = st.selectbox("Imputation method", ["simple_mean", "simple_median", "simple_most_frequent", "grouped_patient_mean", "grouped_age_mean", "grouped_age_ranges_mean", "grouped_gender_mean"])

    df_imputed = task3_impute(df, imputation_method)

    @st.cache
    def task3_dimred(df, dimred_method, sample_size):
        return dim_red(df, dimred_method, sample_size=sample_size, plot=False)

    st.subheader("Clustering")
    col1, col2, col3 = st.columns(3)

    with col1:
        sample_size = st.slider("Sample size", 0.01, 1.0, 0.1)

    with col2:
        dimred_method = st.selectbox("Dimensionality reduction method", ["pacmap", "tsne", "umap"])
    
    with col3:
        cluster_method = st.selectbox("Clustering method", ["--Choose--", "dbscan", "kmeans"])

    df_imputed = task3_dimred(df_imputed, dimred_method, sample_size)

    def task3_cluster(df, cluster_method, dimred_method, sample_size, method_args, selected_feature):
        if selected_feature == "None":
            selected_feature = None

        return cluster(df,
            process_df = False,
            normalize = True,
            algo_name = cluster_method,
            name = "",
            dimred_algo_name=dimred_method,
            dimred_data_name="",
            sample_size=sample_size,
            return_for_app=True,
            selected_label=selected_feature,
            cluster_legend="Cluster",
            **method_args)
    
    if cluster_method == "--Choose--":
        st.write("Please choose a clustering method.")
    else:
        method_args = None
        col1, col2, col3 = st.columns(3)

        if cluster_method == "dbscan":
            with col1:
                eps = st.number_input("eps", value=0.5)
            with col2:
                min_samples = st.number_input("min_samples", min_value=1, value=5)
            method_args = {
                "eps": eps,
                "min_samples": min_samples
            }
        elif cluster_method == "kmeans":
            with col1:
                n_clusters = st.number_input("n_clusters", min_value=0, value=2)
            with col2:
                n_init = st.number_input("n_init", min_value=1, value=10)
            method_args = {
                "n_clusters": n_clusters,
                "n_init": n_init
            }

        with col3:
            all_features = ["None"] + [f for f in df_imputed.columns.tolist() if not f.startswith("DIMREDU_")]
            selected_feature = st.selectbox("Show feature", all_features, index=len(all_features)-2)

        if method_args is not None:
            with st.spinner("clustering..."):
                fig, ax, df_scores = task3_cluster(df_imputed, cluster_method, dimred_method, sample_size, method_args, selected_feature)

                st.write("Results of ", cluster_method)
                col1, col2 = st.columns([1,2])
                with col1:
                    st.write(df_scores.T)
                with col2:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    st.pyplot(fig)


# Task 4
elif main_task_selection == all_tasks[3]:
    st.subheader("Data imputation")

    @st.cache
    def task4_load_dataset(dataset_name):
        # load data set
        df, dfa, dfb = load_data_sets()
        if dataset_name == "A":
            df = dfa
        elif dataset_name == "B":
            df = dfb
        return df

    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.selectbox("Dataset", ["A", "B", "All"])

    df = task4_load_dataset(dataset_name)

    @st.cache
    def task4_impute(df, imputation_method):
        if imputation_method == "simple_mean":
            return imputate_simple(df, "mean")
        elif imputation_method == "simple_median":
            return imputate_simple(df, "median")
        elif imputation_method == "simple_most_frequent":
            return imputate_simple(df, "most_frequent")
        elif imputation_method == "grouped_patient_mean":
            return impute_simple_grouped(df, "mean", "PatientID")
        elif imputation_method == "grouped_age_mean":
            return impute_simple_grouped(df, "mean", "Age")
        elif imputation_method == "grouped_age_ranges_mean":
            return impute_simple_grouped(df, "mean", "Age", age_groups)
        elif imputation_method == "grouped_gender_mean":
            return impute_simple_grouped(df, "mean", "Gender")
        return df
    
    with col2:
        imputation_method = st.selectbox("Imputation method", ["simple_mean", "simple_median", "simple_most_frequent", "grouped_patient_mean", "grouped_age_mean", "grouped_age_ranges_mean", "grouped_gender_mean"])

    df_imputed = task4_impute(df, imputation_method)

    @st.cache
    def task4_dimred(df, dimred_method, sample_size):
        return dim_red(df, dimred_method, sample_size=sample_size, plot=False)

    st.subheader("Clustering")
    col1, col2, col3 = st.columns(3)

    with col1:
        sample_size = st.slider("Sample size", 0.01, 1.0, 0.1)

    with col2:
        dimred_method = st.selectbox("Dimensionality reduction method", ["pacmap", "tsne", "umap"])
    
    with col3:
        cluster_method = st.selectbox("Clustering method", ["--Choose--", "spectral_coclustering", "spectral_biclustering"])

    df_imputed = task4_dimred(df_imputed, dimred_method, sample_size)

    def task4_cluster(df, cluster_method, dimred_method, sample_size, method_args, selected_feature):
        if selected_feature == "None":
            selected_feature = None
            
        return cluster_subspace(df,
            process_df = False,
            normalize = True,
            algo_name = cluster_method,
            name = "",
            dimred_algo_name=dimred_method,
            dimred_data_name="",
            sample_size=sample_size,
            return_for_app=True,
            selected_label=selected_feature,
            cluster_legend="Cluster",
            **method_args)
    
    if cluster_method == "--Choose--":
        st.write("Please choose a clustering method.")
    else:
        method_args = None
        col1, col2, col3 = st.columns(3)

        if cluster_method == "spectral_coclustering":
            with col1:
                n_clusters = st.number_input("n_clusters", min_value=1, value=2)
            with col2:
                n_init = st.number_input("n_init", min_value=1, value=10)
            method_args = {
                "n_clusters": n_clusters,
                "n_init": n_init
            }
        elif cluster_method == "spectral_biclustering":
            with col1:
                n_clusters = st.number_input("n_clusters", min_value=1, value=2)
            with col2:
                n_best = st.number_input("n_best", min_value=1, value=3)
            method_args = {
                "n_clusters": n_clusters,
                "n_best": n_best
            }

        with col3:
            all_features = ["None"] + [f for f in df_imputed.columns.tolist() if not f.startswith("DIMREDU_")]
            selected_feature = st.selectbox("Show feature", all_features, index=len(all_features)-2)

        if method_args is not None:
            with st.spinner("clustering..."):
                fig, ax = task4_cluster(df_imputed, cluster_method, dimred_method, sample_size, method_args, selected_feature)

                st.write("Results of ", cluster_method)
                ax.set_xlabel("")
                ax.set_ylabel("")
                st.pyplot(fig)

    # Subgroups
    #st.subheader("Subgroups")

    #st.write("Searching for subgroups of features that lead to obversations of sepsis.")
    #with st.expander("Show subgroups leading to patients that develop sepsis"):
    #    subgroups_1_5_20 = find_subgroups("SepsisLabel", '1', 5, 20)
    #    st.table(subgroups_1_5_20[["subgroup", "quality"]])
    #with st.expander("Show subgroups leading to patients that do not develop sepsis"):
    #    subgroups_0_5_20 = find_subgroups("SepsisLabel", '0', 5, 20)
    #    st.table(subgroups_0_5_20[["subgroup", "quality"]])

# Task 5
elif main_task_selection == all_tasks[4]:
    st.write("Generation of train and test sets")

    col1, col2, col3 = st.columns(3)

    with col1:
        dataset_name = st.selectbox("Dataset", ["A", "B", "All"])
    
    with col2:
        imputation_method = st.selectbox("Imputation method", ["mean", "median", "most_frequent"])

    @st.cache
    def task5_load_dataset(dataset_name, imputation_method):
        # load data set
        df, dfa, dfb = load_data_sets()
        if dataset_name == "A":
            df = dfa
        elif dataset_name == "B":
            df = dfb

        # imputate
        df = imputate_simple(df, imputation_method)

        # drop na
        df = df.dropna(axis = 1, how = "any")
        df = df.dropna(axis = 0, how = "any")

        return df

    @st.cache
    def task5_sample(df, sample_size, target_label):
        # work on sample size
        df = df.sample(frac=sample_size)

        # get samples values
        X = df.drop(columns = ["SepsisLabel", "Sepsis"])

        # get labels values
        y = df[target_label]

        return X, y

    @st.cache
    def task5_traintest(X, y, test_size):
        # split in train test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

        return X_train, X_test, y_train, y_test

    df = task5_load_dataset(dataset_name, imputation_method)

    with col3:
        sample_size = st.slider("Sample size", 0.01, 1.0, 0.1)

    X, y = task5_sample(df, sample_size, "SepsisLabel")
    X_feature_names = X.columns.tolist()

    # samples stats
    def task5_data_stats(name, X, y):
        with st.expander(name + ": " + str(X.shape[0]) + ", features: " + str(X.shape[1]) + ", labels: " + str(y.shape[0])):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(name + ": ", X.shape[0], ", features: ", X.shape[1])
                if type(X) == pd.DataFrame:
                    st.write(X.describe())
                else:
                    st.write(pd.DataFrame(X).describe())
            with col2:
                st.write("Labels: ", y.shape[0])
                if type(X) == pd.DataFrame:
                    st.write(y.astype({"SepsisLabel": "category"}).describe())
                else:
                    st.write(pd.DataFrame(y).astype("category").describe())
            with col3:
                labels_sepsis = int(y.sum())
                labels_no_sepsis = y.shape[0] - labels_sepsis
                st.write("Sepsis:", labels_sepsis, "No sepsis:", labels_no_sepsis)
                fig, ax = plt.subplots()
                sns.barplot(x=["sepsis", "no_sepsis"], y=[labels_sepsis, labels_no_sepsis])
                st.pyplot(fig)

    task5_data_stats("Samples", X, y)

    # remove dataframe
    X = X.values
    y = np.array(y)

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("Test size", 0.01, 1.0, 0.3)

    X_train, X_test, y_train, y_test = task5_traintest(X, y, test_size)

    # scaling
    def task5_scale(X_train, X_test, scaling_option):
        if scaling_option == "None":
            return X_train, X_test

        if scaling_option == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_option == "MinMaxScaler":
            scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    with col2:
        scaling_option = st.selectbox("Scaling", ["None", "StandardScaler", "MinMaxScaler"])

    with st.spinner("scaling..."):
        X_train, X_test = task5_scale(X_train, X_test, scaling_option)

    # over/undersampling
    def task5_balance(X_train, y_train, balance_option):
        sampler = None
        if balance_option == "Oversampling: SMOTE":
            sampler = SMOTE(n_jobs=-1)
        elif balance_option == "Undersampling: ClusterCentroids":
            sampler = ClusterCentroids()

        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        return X_train, y_train

    with col3:
        balance_option = st.selectbox("Train set balance", ["Imbalanced", "Oversampling: SMOTE", "Undersampling: ClusterCentroids"])

    with st.spinner("balancing..."):
        X_train, y_train = task5_balance(X_train, y_train, balance_option)

    task5_data_stats("Train", X_train, y_train)
    task5_data_stats("Test", X_test, y_test)

    # train
    def task5_train(X_train, y_train, X_test, y_test, algo_name, X_feature_names):
        if algo_name == "SVM":
            classifier = SVC(cache_size=1000)
        elif algo_name == "DT":
            classifier = DecisionTreeClassifier()

        classifier.fit(X_train, y_train)

        # test on train data
        labels = [0, 1]
        label_names = ["no_sepsis", "sepsis"]

        # test on test data
        pred_test = classifier.predict(X_test)
        test_report = pd.DataFrame(classification_report(y_test, pred_test, output_dict=True, labels=labels, target_names=label_names))

        # conf matrix
        conf = confusion_matrix(y_test, pred_test, labels=labels)
        plot = ConfusionMatrixDisplay(conf, display_labels=label_names)

        # export tree
        dot_data = None
        if algo_name == "DT":
            dot_data = export_graphviz(classifier, out_file=None, filled=True, rounded=True, special_characters=True, feature_names=X_feature_names, class_names=label_names)


        return test_report, plot, dot_data

    algo_name = st.selectbox("Classifier algorithm", ["--Choose--", "SVM", "DT"])

    if algo_name == "--Choose--":
        st.write("Please choose a classifier.")
    else:
        st.write("Results of ", algo_name, " classifier:")

        with st.spinner("training..."):
            test_report, conf_plot, dot_data = task5_train(X_train, y_train, X_test, y_test, algo_name, X_feature_names)

            col1, col2 = st.columns(2)
            with col1:
                st.write(test_report)

            with col2:
                conf_plot.plot()
                st.pyplot(conf_plot.figure_)

            if dot_data is not None:
                with st.expander("Show decision tree (warning: large graph)"):
                    st.write("Please wait for graph to load, and scroll down...")
                    st.graphviz_chart(dot_data, use_container_width=True)


# Task 6
elif main_task_selection == all_tasks[5]:
    st.write("Generation of train and test sets")

    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.selectbox("Dataset", ["A", "B", "All"])

    @st.cache
    def task6_load_dataset(dataset_name):
        # load data set
        df, dfa, dfb = load_data_sets()
        if dataset_name == "A":
            df = dfa
        elif dataset_name == "B":
            df = dfb
        return df
    
    df = task6_load_dataset(dataset_name)

    @st.cache
    def task6_create_time_series(df, selected_feature):
        label_class = "SepsisLabel"

        df = df[["ICULOS", "SepsisLabel", selected_feature]]

	    # add patient id back as feature
        df = df.reset_index(level='PatientID')

        # reformat data
        # pyts:
        # univariate: (n_samples, n_timestamps)
        # multivariate: (n_samples, n_features, n_timestamps)
        # TODO there is probably a pandas function to improve this...

        # collect time series and labels per patient
        time_index_max = 0
        time_series_temp = []
        labels = []
        for name, group in df.groupby(by=["PatientID"]):
            #patients.append(name)

            time_index_max = np.max([time_index_max, group["ICULOS"].max()])

            # use last value for target label
            labels.append(group[[label_class]].values[-1][0])

            features = {
                f["ICULOS"]: f[selected_feature]
                for f
                in json.loads(
                    group[["ICULOS", selected_feature]].to_json(orient="records")
                )
            }
            time_series_temp.append(features)

        # reformat features to produce same length vectors
        time_series = []
        for ts in time_series_temp:
            tslist = []
            for i in range(1, time_index_max+1):
                app = False
                if i in ts:
                    if ts[i] is not None:
                        tslist.append(ts[i])
                        app = True
                if not app:
                    tslist.append(np.nan)

            time_series.append(tslist)

        del time_series_temp

        # remove rows and cols with all na, rows need at least 3 (2+1 for label) non na for imputation
        df_time_series = pd.DataFrame(time_series)
        df_time_series["label"] = labels
        df_time_series = df_time_series.dropna(axis = 1, how = "all")
        df_time_series = df_time_series.dropna(axis = 0, how = "all", thresh = 3)
        labels = df_time_series["label"].to_numpy()
        time_series = df_time_series.drop(columns=["label"]).to_numpy()
        del df_time_series
        return labels, time_series

    all_feature_names = sorted(list(set(df.columns)-set(["ICULOS", "SepsisLabel", "Sepsis"])))

    with col2:
        selected_feature = st.selectbox("Feature", all_feature_names)

    labels, time_series = task6_create_time_series(df, selected_feature)

    with st.expander("Time series of feature: " + selected_feature + ", with length of " + str(time_series.shape[1])):
        col1, col2 = st.columns(2)
        
        n_patients = 5
        with col1:
            st.write("Raw data of first", n_patients, "patients:")
            st.write(time_series[:n_patients])

        with col2:
            fig, ax = plt.subplots()
            sns.lineplot(data=np.nan_to_num(time_series, nan=np.inf)[:n_patients].T, sort=False)
            leg = ax.legend(title="Patient")
            leg_lines = leg.get_lines()
            for lind in range(n_patients):
                ax.lines[lind].set_linestyle("-")
                leg_lines[lind].set_linestyle("-")
            ax.set_xlabel("Time")
            ax.set_ylabel("Raw value")
            st.pyplot(fig)

    @st.cache
    def task6_imputate(time_series, labels, imputation_method):
        if imputation_method == "N/A to 0":
            time_series = np.nan_to_num(time_series, nan=0)
        else:
            imputer = InterpolationImputer(strategy=imputation_method)
            time_series = imputer.transform(time_series)

        labels = np.nan_to_num(labels, nan=0)

        return time_series, labels

    imputation_method = st.selectbox("Imputation", ["N/A to 0", "nearest", "linear"])

    time_series, labels = task6_imputate(time_series, labels, imputation_method)

    with st.expander("Imputated time series of feature: " + selected_feature + ", with length of " + str(time_series.shape[1])):
        col1, col2 = st.columns(2)
        
        n_patients = 5
        with col1:
            st.write("Raw data of first", n_patients, "patients:")
            st.write(time_series[:n_patients])

        with col2:
            fig, ax = plt.subplots()
            sns.lineplot(data=time_series[:n_patients, :100].T, sort=False)
            leg = ax.legend(title="Patient")
            leg = ax.legend()
            leg_lines = leg.get_lines()
            for lind in range(n_patients):
                ax.lines[lind].set_linestyle("-")
                leg_lines[lind].set_linestyle("-")
            ax.set_xlabel("Time")
            ax.set_ylabel("Imputed value")
            st.pyplot(fig)
    
    @st.cache
    def task6_traintest(X, y, test_size):
        # split in train test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

        return X_train, X_test, y_train, y_test

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test size", 0.01, 1.0, 0.3)

    X_train, X_test, y_train, y_test = task6_traintest(time_series, labels, test_size)

    with col2:
        scaling_option = st.selectbox("Scaling", ["None", "StandardScaler", "MinMaxScaler"])

    # scaling
    def task6_scale(X_train, X_test, scaling_option):
        if scaling_option == "None":
            return X_train, X_test

        if scaling_option == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_option == "MinMaxScaler":
            scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    with st.spinner("scaling..."):
        X_train, X_test = task6_scale(X_train, X_test, scaling_option)
    
    def task6_data_stats(name, X, y):
        with st.expander(name + ": " + str(X.shape[0]) + ", features: " + str(X.shape[1]) + ", labels: " + str(y.shape[0])):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(name + ": ", X.shape[0], ", features: ", X.shape[1])
                if type(X) == pd.DataFrame:
                    st.write(X.describe())
                else:
                    st.write(pd.DataFrame(X).describe())
            with col2:
                st.write("Labels: ", y.shape[0])
                if type(X) == pd.DataFrame:
                    st.write(y.astype({"SepsisLabel": "category"}).describe())
                else:
                    st.write(pd.DataFrame(y).astype("category").describe())
            with col3:
                labels_sepsis = int(y.sum())
                labels_no_sepsis = y.shape[0] - labels_sepsis
                st.write("Sepsis:", labels_sepsis, "No sepsis:", labels_no_sepsis)
                fig, ax = plt.subplots()
                sns.barplot(x=["sepsis", "no_sepsis"], y=[labels_sepsis, labels_no_sepsis])
                st.pyplot(fig)

    task6_data_stats("Train", X_train, y_train)
    task6_data_stats("Test", X_test, y_test)

    # train
    def task6_train(X_train, y_train, X_test, y_test, algo_name):
        if algo_name == "KNeighborsClassifier":
            classifier = KNeighborsClassifier(n_jobs=-1)
        elif algo_name == "TimeSeriesForest":
            classifier = TimeSeriesForest(n_jobs=-1)
        
        classifier.fit(X_train, y_train)
    
        labels = [0, 1]
        label_names = ["no_sepsis", "sepsis"]

        pred_test = classifier.predict(X_test)
        test_report = pd.DataFrame(classification_report(y_test, pred_test, output_dict=True, labels=labels, target_names=label_names))

        conf = confusion_matrix(y_test, pred_test, labels=labels)
        plot = ConfusionMatrixDisplay(conf, display_labels=label_names)

        return test_report, plot


    algo_name = st.selectbox("Classifier algorithm", ["--Choose--", "KNeighborsClassifier", "TimeSeriesForest"])

    if algo_name == "--Choose--":
        st.write("Please choose a classifier.")
    else:
        st.write("Results of ", algo_name, " classifier:")

        with st.spinner("training..."):
            test_report, conf_plot = task6_train(X_train, y_train, X_test, y_test, algo_name)

            col1, col2 = st.columns(2)
            with col1:
                st.write(test_report)

            with col2:
                conf_plot.plot()
                st.pyplot(conf_plot.figure_)




# if __name__ == "__main__":
#     # https://stackoverflow.com/questions/62760929/how-can-i-run-a-streamlit-app-from-within-a-python-script
#     # https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide
#     import sys
#     import argparse
#     import streamlit.cli
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--main", "-m", action = "store_true")
#     args = parser.parse_args()
#     if args.main:
#         # sys.argv += ["streamlit", "run"]
#         sys.argv = ["streamlit", "run", __file__]
#         sys.exit(st.cli.main())
#         # sys.exit(st.cli.main_run(__file__))
