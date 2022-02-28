import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.cluster import DBSCAN, SpectralCoclustering, SpectralBiclustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from data_challenges.data import load_data_set
from data_challenges.dim_redu import dim_red
from data_challenges.imputation import imputate_simple, impute_simple_grouped


def cluster(df: DataFrame,
            name: str,
            algo_name: str,
            dimred_algo_name: str,
            dimred_data_name: str,
            sample_size: float = 1.0,
            drop_na_rows: bool = False,
            normalize: bool = True,
            process_df: bool = True,
            return_for_app: bool = False,
            selected_label: str = None,
            cluster_legend: str = None,
            **kwargs):
    df_temp = df.copy(deep = True)

    if process_df:
        # drop full NA columns
        # TODO all statt any hier besser?
        df_temp = df.dropna(axis = 1, how = "any")

        # optionally also drop NA rows
        if drop_na_rows:
            df_temp = df.dropna(axis = 0, how = "any")

        # take sample
        # TODO sort
        df_temp = df_temp.sample(frac = sample_size)

    # drop dimred cols for clustering
    dimred_cols = [c for c in list(df_temp.columns) if c.startswith("DIMREDU_")]
    # also drop target labels    
    dimred_cols += ["SepsisLabel", "Sepsis"]

    # scale values, only take "non" sepsis features
    # TODO check scaler need
    if normalize:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_temp.drop(columns = dimred_cols).values)
    else:
        scaled_values = df_temp.drop(columns = dimred_cols).values

    if algo_name == "dbscan":
        algo = DBSCAN(n_jobs = 8, **kwargs)
    elif algo_name == "kmeans":
        # TODO sepsis/gender 1/0 cluster, also try more eg age groups...
        algo = KMeans(**kwargs)

    print("fitting", algo_name)
    clusters = algo.fit(scaled_values)
    labels = clusters.labels_

    if cluster_legend is None:
        cluster_legend = f"cluster_{algo_name}"

    df_temp[f'{cluster_legend}'] = labels

    # Number of clusters in labels, ignoring noise if present.
    clusters_all_count = len(set(labels))
    n_clusters_ = clusters_all_count - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    silhouette = silhouette_score(scaled_values, labels)
    print("Silhouette Coefficient: %0.3f" % silhouette)

    calinski = calinski_harabasz_score(scaled_values, labels)
    print("Calinski and Harabasz score: %0.3f" % calinski)

    davies = davies_bouldin_score(scaled_values, labels)
    print("Davies-Bouldin score: %0.3f" % davies)

    if return_for_app:
        fig, ax = plt.subplots()
    else:
        plt.figure(figsize = (10, 10))
    sns.scatterplot(
            data = df_temp,
            x = f'DIMREDU_{dimred_algo_name}_{dimred_data_name}_sample{sample_size}_0',
            y = f'DIMREDU_{dimred_algo_name}_{dimred_data_name}_sample{sample_size}_1',
            hue = f'{cluster_legend}',
            palette = sns.color_palette('bright', clusters_all_count),
            alpha = 0.5,
            size=selected_label if selected_label is not None else 1
    )
    if not return_for_app:
        plt.title(f"{algo_name} - {name} (#clusters={clusters_all_count}, silhouette={np.around(silhouette, 3)}, calinski={np.around(calinski, 3)}, davies={np.around(davies, 3)})")
        plt.savefig(f"results/vis/{algo_name}-{name}-{dimred_algo_name}-{dimred_data_name}-sample{sample_size}.png", dpi = 600)

    # TODO output params, shapes usw...
    df_scores = DataFrame([{
        "algo_name": algo_name,
        "silhouette": silhouette,
        "calinski": calinski,
        "davies": davies
    }]).T
    if not return_for_app:
        df_scores.to_csv(
            f"results/vis/{algo_name}-{name}-{dimred_algo_name}-{dimred_data_name}-sample{sample_size}.png.meta.csv",
            encoding="UTF-8",
            sep="\t"
        )

    if return_for_app:
        return fig, ax, df_scores


def cluster_subspace(df: DataFrame,
            name: str,
            algo_name: str,
            dimred_algo_name: str,
            dimred_data_name: str,
            sample_size: float = 1.0,
            drop_na_rows: bool = False,
            normalize: bool = True,
            process_df: bool = True,
            return_for_app: bool = False,
            selected_label: str = None,
            cluster_legend: str = None,
            **kwargs):
    df_temp = df.copy(deep = True)

    if process_df:
        # drop full NA columns
        # TODO all statt any hier besser?
        df_temp = df.dropna(axis = 1, how = "any")

        # optionally also drop NA rows
        if drop_na_rows:
            df_temp = df.dropna(axis = 0, how = "any")

        # take sample
        # TODO sort
        df_temp = df_temp.sample(frac = sample_size)

    # drop dimred cols for clustering
    dimred_cols = [c for c in list(df_temp.columns) if c.startswith("DIMREDU_")]
    # also drop target labels    
    dimred_cols += ["SepsisLabel", "Sepsis"]

    # scale values, only take "non" sepsis features
    # TODO check scaler need
    if normalize:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_temp.drop(columns = dimred_cols).values)
    else:
        scaled_values = df_temp.drop(columns = dimred_cols).values

    if algo_name == "spectral_coclustering":
        # 2 clusters: sepsis 1/0
        algo = SpectralCoclustering(**kwargs)
    elif algo_name == "spectral_biclustering":
        algo = SpectralBiclustering(**kwargs)

    clusters = algo.fit(scaled_values)
    labels = clusters.row_labels_

    if cluster_legend is None:
        cluster_legend = f"cluster_{algo_name}"

    df_temp[cluster_legend] = labels

    # Number of clusters in labels, ignoring noise if present.
    clusters_all_count = len(set(labels))
    n_clusters_ = clusters_all_count - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # TODO scores
    # consensus_score

    if return_for_app:
        fig, ax = plt.subplots()
    else:
        plt.figure(figsize = (10, 10))
    sns.scatterplot(
            data = df_temp,
            x = f'DIMREDU_{dimred_algo_name}_{dimred_data_name}_sample{sample_size}_0',
            y = f'DIMREDU_{dimred_algo_name}_{dimred_data_name}_sample{sample_size}_1',
            hue = cluster_legend,
            palette = sns.color_palette('bright', clusters_all_count),
            alpha = 0.5,
            size=selected_label if selected_label is not None else 1
    )
    if not return_for_app:
        plt.title(f"{algo_name} - {name} (#clusters={clusters_all_count})")
        plt.savefig(f"results/vis/{algo_name}-{name}-{dimred_algo_name}-{dimred_data_name}-sample{sample_size}.png", dpi = 300)

    # TODO output params, shapes usw...
    #df_scores = DataFrame([{
    #    "algo_name": algo_name,
    #    "silhouette": silhouette,
    #    "calinski": calinski,
    #    "davies": davies
    #}]).T
    #df_scores.to_csv(
    #    f"results/vis/{algo_name}-{name}-{dimred_algo_name}-{dimred_data_name}-sample{sample_size}.png.meta.csv",
    #    encoding="UTF-8",
    #    sep="\t"
    #)

    if return_for_app:
        return fig, ax


def main_clustering():
    dfa = load_data_set("A", preprocess = True)

    # tsne cant handle NA?
    # dim_red(dfa, "tsne", "", 0.5, True)
    # dim_red(dfa, "pacmap", "", 0.5, drop_na_rows=False)

    # cluster(dfa, 0.01, drop_na_rows=True)

    # simple imputation:
    # univariate, whole dataset without subgroups
    dfa_i_simple_mean = imputate_simple(dfa, "mean")
    # dim_red(dfa_i_simple_mean, "tsne", "i_simple_mean", 0.5)
    # dim_red(dfa_i_simple_mean, "tsne", "i_simple_mean", 1)
    dfa_i_simple_mean_pacmap = dim_red(dfa_i_simple_mean, "pacmap", "i_simple_mean", 0.1)
    # dim_red(dfa_i_simple_mean, "pacmap", "i_simple_mean", 1.0)
    #cluster(dfa_i_simple_mean_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "dbscan",
    #        name = "eps0.5",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_mean",
    #        sample_size=0.5,
    #        eps=0.5)
    #cluster(dfa_i_simple_mean_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "kmeans",
    #        name = "clusters2",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_mean",
    #        sample_size=0.1,
    #        n_clusters=2)
    #cluster(dfa_i_simple_mean_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "kmeans",
    #        name = "clusters5",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_mean",
    #        sample_size=0.1,
    #        n_clusters=5)
    #cluster(dfa_i_simple_mean_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "kmeans",
    #        name = "clusters10",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_mean",
    #        sample_size=0.1,
    #        n_clusters=10)
    cluster_subspace(dfa_i_simple_mean_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters2",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_mean",
            sample_size=0.1,
            n_clusters=2)
    cluster_subspace(dfa_i_simple_mean_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters5",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_mean",
            sample_size=0.1,
            n_clusters=5)
    cluster_subspace(dfa_i_simple_mean_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters10",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_mean",
            sample_size=0.1,
            n_clusters=10)

    # saving is ~500mb
    # dfa_i_simple_mean.to_csv("data/imputations/train_a_imputation_simple_mean.csv", sep = '|')
    
    
    dfa_i_simple_median = imputate_simple(dfa, "median")
    dfa_i_simple_median_pacmap = dim_red(dfa_i_simple_median, "pacmap", "i_simple_median", 0.1)
    #cluster(dfa_i_simple_median_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "dbscan",
    #        name = "eps0.5",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_median",
    #        sample_size=0.1,
    #        eps=0.5)
    #cluster(dfa_i_simple_median_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "kmeans",
    #        name = "clusters2",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_median",
    #        sample_size=0.1,
    #        n_clusters=2)
    cluster_subspace(dfa_i_simple_median_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters2",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_median",
            sample_size=0.1,
            n_clusters=2)
    cluster_subspace(dfa_i_simple_median_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters5",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_median",
            sample_size=0.1,
            n_clusters=5)
    cluster_subspace(dfa_i_simple_median_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters10",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_median",
            sample_size=0.1,
            n_clusters=10)


    dfa_i_simple_most_frequent = imputate_simple(dfa, "most_frequent")
    dfa_i_simple_most_frequent_pacmap = dim_red(dfa_i_simple_most_frequent, "pacmap", "i_simple_most_frequent", 0.1)
    #cluster(dfa_i_simple_most_frequent_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "dbscan",
    #        name = "eps0.5",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_most_frequent",
    #        sample_size=0.1,
    #        eps=0.5)
    #cluster(dfa_i_simple_most_frequent_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "kmeans",
    #        name = "clusters2",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_most_frequent",
    #        sample_size=0.1,
    #        n_clusters=2)
    cluster_subspace(dfa_i_simple_most_frequent_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters2",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_most_frequent",
            sample_size=0.1,
            n_clusters=2)
    cluster_subspace(dfa_i_simple_most_frequent_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters5",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_most_frequent",
            sample_size=0.1,
            n_clusters=5)
    cluster_subspace(dfa_i_simple_most_frequent_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters10",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_most_frequent",
            sample_size=0.1,
            n_clusters=10)


    # dim_red(dfa_i_simple_most_frequent, "pacmap", "i_simple_most_frequent", 0.5)
    # dim_red(dfa_i_simple_most_frequent, "pacmap", "i_simple_most_frequent", 1.0)

    dfa_i_simple_group_gender_mean = impute_simple_grouped(dfa, "mean", "Gender")
    dfa_i_simple_group_gender_mean_pacmap = dim_red(dfa_i_simple_group_gender_mean, "pacmap", "i_simple_group_gender_mean", 0.1)
    #cluster(dfa_i_simple_group_gender_mean_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "dbscan",
    #        name = "eps0.5",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_group_gender_mean",
    #        sample_size=0.1,
    #        eps=0.5)
    #cluster(dfa_i_simple_group_gender_mean_pacmap,
    #        process_df = False,
    #        normalize = True,
    #        algo_name = "kmeans",
    #        name = "clusters2",
    #        dimred_algo_name="pacmap",
    #        dimred_data_name="i_simple_group_gender_mean",
    #        sample_size=0.1,
    #        n_clusters=2)
    cluster_subspace(dfa_i_simple_group_gender_mean_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters2",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_group_gender_mean",
            sample_size=0.1,
            n_clusters=2)
    cluster_subspace(dfa_i_simple_group_gender_mean_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters5",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_group_gender_mean",
            sample_size=0.1,
            n_clusters=5)
    cluster_subspace(dfa_i_simple_group_gender_mean_pacmap,
            process_df = False,
            normalize = True,
            algo_name = "spectral_coclustering",
            name = "clusters10",
            dimred_algo_name="pacmap",
            dimred_data_name="i_simple_group_gender_mean",
            sample_size=0.1,
            n_clusters=10)

    print("done")


if __name__ == "__main__":
    main_clustering()
