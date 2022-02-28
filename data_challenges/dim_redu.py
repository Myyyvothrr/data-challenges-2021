import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pacmap import PaCMAP
from umap import UMAP
import umap.plot
from pandas import DataFrame
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Any


def dim_red(df: DataFrame,
            algo_name: str,
            data_name: str = "",
            sample_size: float = 1,
            drop_na_rows: bool = False,
            n_dims: int = 2,
            normalize: bool = True,
            plot: bool = True,
            save_plot: bool = True,
            return_plot: bool = False,
            plot_hue_variable: str = "SepsisLabel",
            return_for_app: bool = False,
            **kwargs) -> Any:
    """
    This function performs dimensionality reduction on the data.

    Args:
        df:
        algo_name: one of [pacmap, umap, tsne]
        data_name: name of the output image (the algo_name will be prepended & the sample_size will be appended)
        sample_size: frac of the df to use
        drop_na_rows: drop rows where all values are NA
        n_dims: number of dimensions to reduce to
        normalize: apply StandardScaler (from sklearn.preprocessing)
        plot:
        kwargs: Pass additional arguments to the algorithm
    """
    # drop full NA columns
    # DONE: all statt any hier? => yes
    df_temp = df.dropna(axis = 1, how = "all")

    # optionally also drop NA rows
    if drop_na_rows:
        df_temp = df.dropna(axis = 0, how = "all")

    # take sample
    # TODO sort
    df_temp = df_temp.sample(frac = sample_size)

    # scale values, only take "non" sepsis features
    # TODO check scaler need
    if normalize:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_temp.drop(columns = ["SepsisLabel", "Sepsis"]).values)
    else:
        scaled_values = df_temp.drop(columns=["SepsisLabel", "Sepsis"]).values

    if algo_name.lower() == "tsne":
        # TODO params
        algo = TSNE(verbose = 1,
                    n_iter = 250,
                    n_components = n_dims,
                    n_jobs = 8,
                    **kwargs)
    elif algo_name.lower() == "pacmap":
        # https://github.com/YingfanWang/PaCMAP#parameters
        algo = PaCMAP(verbose = True,
                      num_iters = 250,
                      n_dims = n_dims,
                      n_neighbors = None,
                      **kwargs)
    elif algo_name.lower() == "umap":
        # https://umap-learn.readthedocs.io/en/latest/api.html
        algo = UMAP(verbose = True,
                    n_components = n_dims,
                    # n_neighbors = 15,
                    # n_epochs = None,
                    **kwargs)
    else:
        raise ValueError("Algorithm not supported")

    embedding = algo.fit_transform(scaled_values)
    for dim in range(embedding.shape[1]):
        # df_temp[f'dim-{dim}'] = embedding[:, dim]
        df_temp[f'DIMREDU_{algo_name}_{data_name}_sample{sample_size}_{dim}'] = embedding[:, dim]

    if plot:
        # get labels of the embedding
        dimredu_labels = [x for x in df_temp.columns.tolist() if x.startswith("DIMREDU_")]
        dimredu_labels = [
                [x for x in df_temp.columns.tolist() if x.startswith(dimredu_label)]
                for dimredu_label in
                set(["_".join(x.split("_")[:-1]) for x in dimredu_labels])
        ]
        dimredu_label = dimredu_labels[0]

        if return_for_app:
            return df_temp, dimredu_label

        if embedding.shape[1] == 2:
            if return_plot:
                fig, ax = plt.subplots()
            else:
                plt.figure(figsize = (10, 10))
            sns.scatterplot(
                    data = df_temp,
                    x = dimredu_label[0],  # "dim-1",
                    y = dimredu_label[1],  # "dim-2",
                    hue = plot_hue_variable,
                    palette = sns.color_palette('bright', 2),
                    alpha = 0.5
            )
            plt.title(algo_name)
            if save_plot:
                plt.savefig(f"results/vis/{algo_name}-{data_name}-sample{sample_size}.png", dpi = 300)
            if return_plot:
                return fig, ax
        elif embedding.shape[1] == 3:
            fig = px.scatter_3d(
                    df_temp,
                    x = dimredu_label[0],
                    y = dimredu_label[1],
                    z = dimredu_label[2],
                    color = "SepsisLabel",
                    labels = {'color': 'SepsisLabel'},
            )

            fig.show()

    return df_temp
