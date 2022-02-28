import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime
from os import makedirs
from pyts.classification import KNeighborsClassifier, TimeSeriesForest
from pyts.preprocessing import InterpolationImputer, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from data_challenges.data import load_data_set


def train(algo, meta_data):
	algo_short_name = ""
	if algo == "KNeighborsClassifier":
		algo_short_name = "KNN"
		classifier = KNeighborsClassifier(n_jobs=-1)
	elif algo == "TimeSeriesForest":
		algo_short_name = "TSF"
		classifier = TimeSeriesForest(n_jobs=-1)

	classifier.fit(X_train, y_train)
	joblib.dump(classifier, f"{results_dir}/classifier_{algo}.joblib")

	labels = [0, 1]
	label_names = ["no_sepsis", "sepsis"]

	# test on train data
	pred_train = classifier.predict(X_train)
	meta_data[f"train_results_{algo}"] = classification_report(y_train, pred_train, output_dict=True, labels=labels, target_names=label_names)

	# test on test data
	pred_test = classifier.predict(X_test)
	test_report = classification_report(y_test, pred_test, output_dict=True, labels=labels, target_names=label_names)
	meta_data[f"test_results_{algo}"] = test_report

	conf = confusion_matrix(y_test, pred_test, labels=labels)
	plot = ConfusionMatrixDisplay(conf, display_labels=label_names)
	plot.plot()
	plt.title(f"{algo} test")
	plt.savefig(f"{results_dir}/confusion_matrix_{algo}_test.png", dpi=600, bbox_inches="tight")
	plt.savefig(f"{results_dir}/confusion_matrix_{algo}_test_300.png", dpi=300, bbox_inches="tight")
	plt.savefig(f"{results_dir}/confusion_matrix_{algo}_test_72.png", dpi=72, bbox_inches="tight")

	meta_data[f"test_results_{algo}_latex_all"] = f"\textbf{{{algo_short_name}}} & ${test_report['accuracy']:.3f}$ & ${test_report['macro avg']['f1-score']:.3f}$ & ${test_report['macro avg']['precision']:.3f}$ & ${test_report['macro avg']['recall']:.3f}$ & ${test_report['weighted avg']['f1-score']:.3f}$ & ${test_report['weighted avg']['precision']:.3f}$ & ${test_report['weighted avg']['recall']:.3f}$ \\"
	meta_data[f"test_results_{algo}_latex_per_class"] = f"\textbf{{{algo_short_name}}} & ${test_report['no_sepsis']['f1-score']:.3f}$ & ${test_report['no_sepsis']['precision']:.3f}$ & ${test_report['no_sepsis']['recall']:.3f}$ & ${test_report['sepsis']['f1-score']:.3f}$ & ${test_report['sepsis']['precision']:.3f}$ & ${test_report['sepsis']['recall']:.3f}$ \\"


if __name__ == '__main__':
	meta_data = {
		# dataset to use
		"dataset_name": "A",

		# label class to use
		"label_class": "SepsisLabel",

		# imputation method
		"imputation_method": "linear",

		# selected feature
		"feature": "HR",

		# size of test set
		"test_size": 0.3,
	}

	now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
	results_dir = f"results/time_series/{meta_data['dataset_name']}/{now}-data_{meta_data['dataset_name']}-label_{meta_data['label_class']}-feature_{meta_data['feature']}-imputation_{meta_data['imputation_method']}"
	makedirs(results_dir, exist_ok=True)
	meta_data["results_dir"] = results_dir

	# load dataset
	df = load_data_set(meta_data['dataset_name'], preprocess = True)

	# extract features
	df = df[["ICULOS", meta_data["label_class"], meta_data["feature"]]]
	
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
	for name, group in tqdm(df.groupby(by=["PatientID"])):
		#patients.append(name)

		time_index_max = np.max([time_index_max, group["ICULOS"].max()])

		# use last value for target label
		labels.append(group[[meta_data["label_class"]]].values[-1][0])

		features = {
			f["ICULOS"]: f[meta_data["feature"]]
			for f
			in json.loads(
				group[["ICULOS", meta_data["feature"]]].to_json(orient="records")
			)
		}
		time_series_temp.append(features)

	# reformat features to produce same length vectors
	time_series = []
	for ts in tqdm(time_series_temp):
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

	meta_data["time_series_shape"] = time_series.shape
	meta_data["labels_shape"] = labels.shape

	# plot first 5
	#df_time_series = pd.DataFrame(time_series)
	#sns.lineplot(x="x", y="y", sort=False, data=df_time_series);

	# impute data
	# TODO fehler warum?
	imputer = InterpolationImputer(strategy=meta_data["imputation_method"])
	time_series = imputer.transform(time_series)
	#time_series = np.nan_to_num(time_series, nan=0)
	# TODO labels?
	labels = np.nan_to_num(labels, nan=0)

	# split train test
	X_train, X_test, y_train, y_test = train_test_split(time_series, labels, test_size=0.3, shuffle=True)
	meta_data["x_train_shape"] = X_train.shape
	meta_data["y_train_shape"] = y_train.shape
	meta_data["X_test_shape"] = X_test.shape
	meta_data["y_test_shape"] = y_test.shape

	# scale
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	joblib.dump(scaler, f"{results_dir}/scaler.joblib")

	# train classifers
	train("KNeighborsClassifier", meta_data)
	train("TimeSeriesForest", meta_data)

	# save results
	with open(f"{results_dir}/results.json", "w", encoding="UTF-8") as json_file:
		json.dump(meta_data, json_file, indent=2, ensure_ascii=False)
