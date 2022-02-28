import sys
from pathlib import Path
import streamlit as st
import streamlit.cli


if __name__ == "__main__":
    # https://stackoverflow.com/questions/62760929/how-can-i-run-a-streamlit-app-from-within-a-python-script
    # https://stackoverflow.com/questions/60172282/how-to-run-debug-a-streamlit-application-from-an-ide

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--main", "-m", action = "store_true")
    # args = parser.parse_args()
    # if args.main:

    sys.argv = ["streamlit", "run", Path(__file__).parent.joinpath("app.py").as_posix()]
    sys.exit(st.cli.main())
    # sys.exit(st.cli.main_run(__file__))
