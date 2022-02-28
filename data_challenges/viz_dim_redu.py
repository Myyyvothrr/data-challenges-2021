from pathlib import Path
import bokeh
import bokeh.io
import bokeh.models
import bokeh.plotting
import bokeh.layouts
import bokeh.server.server
import bokeh.themes
import bokeh.transform
import numpy as np
import pandas as pd
from functools import partial
from data_challenges.data import PATHs
from typing import List, Tuple
from pandas import DataFrame


def get_data(df = None,
             parquet_path = None,
             default_feat = "SepsisLabel",
             sample_size: float = None,
             ) -> Tuple[DataFrame, str, List[str]]:
    # Get df
    if df is None:
        try:
            df_dimredu = pd.read_parquet(parquet_path)
        except:
            print(f'\nCould not load: {parquet_path}!\n')
            results_p = PATHs.results.joinpath("dim_redus")
            df_dimredu = pd.read_parquet(list(results_p.glob("**/*.parquet"))[0])
    # else:
    #     raise Exception("Need to specify either df or parquet_path")
    if sample_size is not None:
        df_dimredu = df_dimredu.sample(frac = sample_size)

    # Get feature names
    dimredu_labels = [x for x in df_dimredu.columns.tolist() if x.upper().startswith("DIMREDU")]
    dimredu_labels = [
        [x for x in df_dimredu.columns.tolist() if x.startswith(dimredu_label)]
        for dimredu_label in
        set(["_".join(x.split("_")[:-1]) for x in dimredu_labels])
    ]
    dimredu_label = dimredu_labels[0]

    return df_dimredu, default_feat, dimredu_label


def _bokeh_change_color_by_feat(doc):
    """
    Plot Dimensionality Reduction with Bokeh.
    Allows you to select which features to use for color encoding.

    Examples:
        >>> bokeh_change_color_by_feat()
    """
    global df_dimredu, default_feat, dimredu_label

    p = bokeh.plotting.figure(
        title = "Dimensionality reduction",
        background_fill_color = "#fafafa",
        tools = "crosshair, pan, reset, save, wheel_zoom, box_zoom, lasso_select, poly_select, tap, undo, redo, save, box_select, hover",
        width = 1100,
        height = 800,
    )
    # p.toolbar.active_inspect = [hover_tool, crosshair_tool]
    # p.xaxis.axis_label = 'Flipper Length (mm)'
    # p.yaxis.axis_label = 'Body Mass (g)'

    cmap = bokeh.transform.linear_cmap(default_feat,
                                       "Inferno256",  # Inferno256 Category10_3
                                       low = min(df_dimredu[default_feat]),
                                       high = max(df_dimredu[default_feat]),
                                       )

    points = p.scatter(
        x = dimredu_label[0], y = dimredu_label[1],
        source = df_dimredu,
        # legend_group = "Gender",
        fill_alpha = 0.2,
        size = 9,
        # marker = factor_mark('species', MARKERS, SPECIES),
        # color = bokeh.transform.factor_cmap(default_feat, 'Category10_3', sorted(df_dimredu[default_feat].unique()), high = 1),
        color = cmap,
        # color = bokeh.transform.linear_cmap(default_feat, "Category10_3", low = 0, high = 1),
    )

    cbar = bokeh.models.ColorBar(color_mapper = cmap.get("transform"), location = (0, 0))
    p.add_layout(cbar, "right")

    # def bokeh_update_color_button(event):
    #     old = points.glyph.fill_color["field"]
    #     options = df_dimredu.columns.tolist()
    #     new = options[(options.index(old) + 1) % len(options)]
    #     select.trigger("value", old, new)
    #     # bokeh_update_color(None, old, new)

    def bokeh_update_color(attr, old, new):
        # color_mapper = bokeh.models.LinearColorMapper(
        #                     palette = "Inferno256",  # Category10_3 Inferno256
        #                     low = min(df_dimredu[new]),
        #                     high = max(df_dimredu[new])
        #                     )
        # color_mapper = bokeh.transform.linear_cmap(
        #                     new,
        #                     palette = "Inferno256",  # Category10_3 Inferno256
        #                     low = min(df_dimredu[new]),
        #                     high = max(df_dimredu[new])
        #                     )
        cmap["field"] = new
        cmap["transform"].low = min(df_dimredu[new])
        cmap["transform"].high = max(df_dimredu[new])
        # color_mapper = bokeh.transform.linear_cmap(new, "Category10_3", low = min(df_dimredu[new]), high = max(df_dimredu[new]))
        # cbar = bokeh.models.ColorBar(color_mapper = color_mapper, location = (0, 0))
        # p.add_layout(cbar, "left")
        # cbar.color_mapper = color_mapper.get("transform")
        points.glyph.fill_color = cmap  # {"field": new, "transform": color_mapper}
        points.glyph.line_color = cmap  # {"field": new, "transform": color_mapper}

    select = bokeh.models.Select(
        title = "Features",
        value = default_feat,
        options = df_dimredu.columns.tolist(),  # [::6],
        width = 100,
    )
    # select.js_link("value", points.glyph, "fill_color", "field")
    # select.js_link("value", points.glyph, "fill_color")
    select.on_change("value", bokeh_update_color)

    # button = bokeh.models.Button(label = "Feature", button_type = "success")
    # button.on_event("click", bokeh_update_color_button)

    # p.legend.location = "top_left"
    # p.legendend.title = default_feat


# def _bokeh_change_color_by_feat_server(doc):
    # doc.add_root(bokeh.layouts.column(bokeh.layouts.row(select, button), p))
    doc.add_root(bokeh.layouts.column(select, p))
    # bokeh.plotting.show(bokeh.layouts.column(select, p))

    doc.theme = bokeh.themes.built_in_themes[bokeh.themes.CALIBER]


def bokeh_change_color_by_feat(df = None,
                               parquet_path = None,
                               default_feature = "SepsisLabel",
                               sample_size: float = None,
                               ):
    """
    Run a server for _bokeh_change_color_by_feat().
    Needed for python callbacks.

    Examples:
        >>> bokeh_change_color_by_feat()
        >>> bokeh_change_color_by_feat(
        >>>     parquet_path = list(PATHs.results.joinpath("dim_redus").glob("**/*-all.parquet"))[0],
        >>>     default_feature = "SepsisLabel",
        >>>     sample_size = .01
        >>> )
    """
    global df_dimredu, default_feat, dimredu_label
    if parquet_path is not None:
        print(f'Loading data from: {parquet_path}')
    df_dimredu, default_feat, dimredu_label = get_data(
            df = df,
            parquet_path = parquet_path,
            default_feat = default_feature,
            sample_size = sample_size,
    )

    server = bokeh.server.server.Server(
        _bokeh_change_color_by_feat,
        # num_procs = 4,
    )
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
    # bokeh.plotting.show(_bokeh_change_color_by_feat)


def ___create_sample_dfs():
    p = list(PATHs.results.joinpath("dim_redus").glob("**/*-all.parquet"))[0]
    p = list(PATHs.results.joinpath("dim_redus").glob("**/PaCMAP_imputed-iter-60-mean-asc-all.parquet"))[0]
    # p = list(PATHs.results.joinpath("dim_redus").glob("**/dimredu_SepsisLabel1_imputed-iter-60-mean-asc-all_PaCMAP.parquet"))[0]
    # p = Path("PaCMAP_imputed-iter-60-mean-asc-all.parquet")
    frac = 0.01
    pd.read_parquet(p).sample(frac = frac).to_parquet(p.with_name(p.stem + f"-sample{frac}.parquet"))


if __name__ == "__main__":
    raise Exception("Deprecated. Use viz_dim_redu_js.py instead.")
    # _bokeh_change_color_by_feat()
    # bokeh_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP-A-group-mean-impute-mean.parquet"))
    # bokeh_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP_imputed-iter-60-mean-asc-all.parquet"))
    # ___create_sample_dfs()
    # bokeh_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP_imputed-iter-60-mean-asc-all-sample0.05.parquet"), sample_size = 0.2)
    # bokeh_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP_imputed-iter-60-mean-asc-all-sample0.01.parquet"))
    # bokeh_change_color_by_feat(
    #     parquet_path = list(PATHs.results.joinpath("dim_redus").glob("**/*-all.parquet"))[0],
    #     default_feature = "SepsisLabel",
    #     sample_size = .01,
    # )
