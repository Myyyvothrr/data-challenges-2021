from pathlib import Path
import sys
import os
from typing import Union, Optional
from os import PathLike
from data_challenges.data import PATHs
import pandas as pd


def is_ipython() -> bool:
    """
    Check if calling from IPython.

    - https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
    - https://newbedev.com/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/
    """
    try:
        get_ipython = sys.modules['IPython'].get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            raise ImportError("console")
        # if 'VSCODE_PID' in os.environ:
        #     raise ImportError("vscode")
    except Exception:
        # using Python
        return False
    else:
        # using IPython
        return True


def results_list(path: Union[str, PathLike] = PATHs.results, glob: str = "*.png") -> list:
    """
    Lists all assets in a directory recursively using glob.

    Args:
        path: From where to look recursively for glob
    """

    return list(Path(path).rglob(glob))


def pandas_set_display_no_nan(self):
    # pd.set_option("display.max_rows", 200)
    pd.options.display.max_columns = 50
    # pd.options.display.precision = 6

    # overwrite the df _repr_html_ to render nan values as "܁"
    if not hasattr(pd.DataFrame, "__repr_html_bak"):
        # https://github.com/pandas-dev/pandas/blob/2e29e1172bb5d17c5d6f4d8bec1d3e6452091822/pandas/core/frame.py#L1001
        pd.DataFrame.__repr_html_bak = pd.DataFrame._repr_html_
    def df_html_no_nan(self):
        # ·⸱⸱܁ ▉ # https://www.compart.com/de/unicode/U+0701 # https://en.wikipedia.org/wiki/Box-drawing_character
        # self = self.fillna("܁")  # fillna("܁") can sometimes produce errors
        html = pd.DataFrame.__repr_html_bak(self)
        # return html
        return html.replace("<td>NaN</td>", "<td>܁</td>").replace(r"<td>&lt;NA&gt;</td>", "<td>܁</td>")
        return html.replace("NaN", "܁").replace("<NA>", "܁")
    pd.DataFrame._repr_html_ = df_html_no_nan


def pandas_set_auto_style():
    """
    Sets pd.DataFrame.style_auto() method.
    """
    def df_auto_style(self,
                      precision: Optional[int] = None,
                      na_rep: Optional[str] = "܁",
                      thousands: Optional[str] = ",",
                      subset = None,
                      ):
        # self.reset_index(inplace = True)
        # style = DataFrame([[True, False], [1,2,5], [1e-5, 1e-12, 1e-22], [np.pi, np.pi * 1e-3, np.pi * 1e-6], [1e22, 1e-22], [None, np.nan], ["text", "lol"]]).T \
        style = self.style.background_gradient(
                subset = subset,
            ).bar(
                subset = subset,
            ).highlight_null(
                subset = subset,
            ).set_properties(
                subset = subset,
                **{
                    # "border-style": "solid", "border-color": "black",
                    # "background-color": "lightgrey",
                }
            # ).set_table_styles(
            #     [
            #         {'selector': 'th', 'props': [('font-size', '10pt')]},
            #         {'selector': 'td', 'props': [('font-size', '10pt')]},
            #         {'selector': 'table', 'props': [('border-style', 'solid')]},
            #         {'selector': 'th', 'props': [('border-style', 'solid')]},
            #         {'selector': 'td', 'props': [('border-style', 'solid')]},
            #     ]
            # ).format(
            #     na_rep = na_rep,
            #     thousands = thousands,
            #     precision = precision
            ).export()
        return self.style.use(style).format(
                na_rep = na_rep,
                thousands = thousands,
                precision = precision,
                subset = subset,
            ).apply(
                lambda s: ["color: green; background-color: rgba(0,255,0,0.05);" if x is True else "color: red; background-color: rgba(255,0,0,0.2)" if x is False else "" for x in s],
                subset = subset,
            )
    # if hasattr(pd.DataFrame, "style_auto"):
    #     raise Exception("style_auto already exists")
    # else:
    # if not hasattr(pd.DataFrame, "style_auto") or input("Do you want to overwrite the method style_auto? (y/n):") == "y":
    pd.DataFrame.style_auto = df_auto_style
