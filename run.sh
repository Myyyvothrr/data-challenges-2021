# run web app
# fixes error with categorical features, see https://discuss.streamlit.io/t/after-upgrade-to-the-latest-version-now-this-error-id-showing-up-arrowinvalid/15794/24
streamlit run app.py --global.dataFrameSerialization="legacy"
