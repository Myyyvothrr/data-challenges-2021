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
             sample_size: float = None,
             ) -> Tuple[DataFrame, List[str]]:
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

    return df_dimredu, dimredu_label


def bokeh_viz_change_color_by_feat(df = None,
                                   parquet_path = None,
                                   default_feat = "Sepsis",
                                   sample_size: float = None,
                                   ):
    """
    Plot Dimensionality Reduction with Bokeh.
    Allows you to select which features to use for color encoding.

    Examples:
        >>> bokeh_viz_change_color_by_feat()
    """
    if parquet_path is not None:
        print(f'Loading data from: {parquet_path}')
    df_dimredu, dimredu_label = get_data(
            df = df,
            parquet_path = parquet_path,
            sample_size = sample_size,
    )

    p = bokeh.plotting.figure(
            title = "Dimensionality reduction",
            background_fill_color = "#fafafa",
            tools = "crosshair, pan, reset, save, wheel_zoom, box_zoom, lasso_select, poly_select, tap, undo, redo, "
                    "save, box_select, hover",
            active_scroll ="wheel_zoom",
            width = 1100,
            height = 800,
    )

    cmap = bokeh.transform.linear_cmap(default_feat,
                                       "Inferno256",  # Inferno256 Category10_3
                                       low = min(df_dimredu[default_feat]),
                                       high = max(df_dimredu[default_feat]),
                                       )

    points = p.scatter(
            x = dimredu_label[0], y = dimredu_label[1],
            source = df_dimredu,
            fill_alpha = 0.2,
            size = 9,
            # color = bokeh.transform.factor_cmap(default_feat, 'Category10_3', sorted(df_dimredu[default_feat].unique()), high = 1),
            color = cmap,
            # color = bokeh.transform.linear_cmap(default_feat, "Category10_3", low = 0, high = 1),
    )

    cbar = bokeh.models.ColorBar(color_mapper = cmap.get("transform"), location = (0, 0))
    p.add_layout(cbar, "right")

    js_callback = bokeh.models.CustomJS(
            args = dict(
                    source = bokeh.models.ColumnDataSource(df_dimredu),
                    # source = bokeh.models.ColumnDataSource((df_dimredu - df_dimredu.min(axis = 0)) / (df_dimredu.max(axis = 0) - df_dimredu.min(axis = 0))),
                    points = points,
                    cmap = cmap),
            code = """
            const df = source.data;
            // console.log("points", points);
            // console.log("cb_obj", cb_obj);
            // console.log("cmap", cmap);
            // console.log("source", source);
            // console.log("df", df);
            // console.log("new values!", df[cb_obj.value]);
            // console.log("df.data", df.data);
            // console.log("min", Math.min(...df[cb_obj.value]));
            // console.log("max", Math.max(...df[cb_obj.value]));

            // console.log("cmap.field", cmap.field);
            cmap.field = cb_obj.value;
            cmap.transform.low = Math.min(...df[cb_obj.value]);
            cmap.transform.high = Math.max(...df[cb_obj.value]);


            // df["color"] = df[cb_obj.value]
            // const x = df[cb_obj.value]["x"];
            // const y = df[cb_obj.value]["y"];
            // console.log("x", x);
            // console.log("y", y);

            // console.log("points.glyph.fill_color", points.glyph.fill_color);
            points.glyph.fill_color = cmap;
            points.glyph.line_color = cmap;

            cmap.transform.change.emit();
            // source.change.emit();
            points.change.emit();
            """
    )

    select = bokeh.models.Select(
            title = "Features",
            value = default_feat,
            options = df_dimredu.columns.tolist(),  # [::6],
            width = 100,
    )

    # select.js_link("value", points.glyph, "fill_color", "field")
    # select.js_link("value", points.glyph, "fill_color")
    select.js_on_change("value", js_callback)
    bokeh.plotting.output_file(PATHs.results_dimred.joinpath("viz_dimred_interactive.html"))
    bokeh.plotting.show(bokeh.layouts.column(select, p))


if __name__ == "__main__":
    # _bokeh_change_color_by_feat()
    # bokeh_viz_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP-A-group-mean-impute-mean.parquet"))
    # bokeh_viz_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP_imputed-iter-60-mean-asc-all.parquet"))
    # bokeh_viz_change_color_by_feat(parquet_path = PATHs.results_dimred.joinpath("PaCMAP_imputed-iter-60-mean-asc-all-sample0.05.parquet"), sample_size = 0.2)
    bokeh_viz_change_color_by_feat(
            parquet_path = PATHs.results_dimred.joinpath("PaCMAP_imputed-iter-60-mean-asc-all-sample0.01.parquet"),
            default_feat = "Sepsis")
    # bokeh_viz_change_color_by_feat(
    #     parquet_path = list(PATHs.results.joinpath("dim_redus").glob("**/*-all.parquet"))[0],
    #     default_feature = "SepsisLabel",
    #     sample_size = .01,
    # )
