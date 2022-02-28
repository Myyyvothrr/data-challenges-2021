import sys
sys.path.append(".")

import graphviz
import joblib
import json
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from os import makedirs

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from data_challenges.data import load_data_set
from data_challenges.imputation import imputate_simple


def train(X_train, y_train, X_test, y_test, X_feature_names, name, algo_name, results_dir):
	meta_data = {}

	# train svm classifier
	algo_short_name = ""
	if algo_name == "svm":
		algo_short_name = "SVM"
		classifier = SVC(cache_size=1000)
	elif algo_name == "decision_tree":
		algo_short_name = "DT"
		classifier = DecisionTreeClassifier()
	else:
		print("unknown classifier:", algo_name)
		exit(1)

	print("fitting classifier...")
	classifier.fit(X_train, y_train)
	joblib.dump(classifier, f"{results_dir}/classifier_{name}.joblib")

	# test on train data
	labels = [0, 1]
	label_names = ["no_sepsis", "sepsis"]

	print("evaluating classifier...")
	pred_train = classifier.predict(X_train)
	meta_data[f"train_results_{name}"] = classification_report(y_train, pred_train, output_dict=True, labels=labels, target_names=label_names)

	# test on test data
	pred_test = classifier.predict(X_test)
	test_report = classification_report(y_test, pred_test, output_dict=True, labels=labels, target_names=label_names)
	meta_data[f"test_results_{name}"] = test_report

	# export tree
	if algo_name == "decision_tree":
		dot_data = export_graphviz(classifier, out_file=None, filled=True, rounded=True, special_characters=True, feature_names=X_feature_names, class_names=label_names) 
		graph = graphviz.Source(dot_data)
		graph.render(f"{results_dir}/decision_tree_{name}")

	# conf matrix
	conf = confusion_matrix(y_test, pred_test, labels=labels)
	plot = ConfusionMatrixDisplay(conf, display_labels=label_names)
	plot.plot()
	plt.title(name + " test")
	plt.savefig(f"{results_dir}/confusion_matrix_test_{name}.png", dpi=600, bbox_inches="tight")
	plt.savefig(f"{results_dir}/confusion_matrix_test_{name}_300.png", dpi=300, bbox_inches="tight")
	plt.savefig(f"{results_dir}/confusion_matrix_test_{name}_72.png", dpi=72, bbox_inches="tight")

	# export latex table
	meta_data[f"test_results_{name}_latex_all"] = f"\textbf{{{algo_short_name}}} & ${test_report['accuracy']:.3f}$ & ${test_report['macro avg']['f1-score']:.3f}$ & ${test_report['macro avg']['precision']:.3f}$ & ${test_report['macro avg']['recall']:.3f}$ & ${test_report['weighted avg']['f1-score']:.3f}$ & ${test_report['weighted avg']['precision']:.3f}$ & ${test_report['weighted avg']['recall']:.3f}$ \\"
	meta_data[f"test_results_{name}_latex_per_class"] = f"\textbf{{{algo_short_name}}} & ${test_report['no_sepsis']['f1-score']:.3f}$ & ${test_report['no_sepsis']['precision']:.3f}$ & ${test_report['no_sepsis']['recall']:.3f}$ & ${test_report['sepsis']['f1-score']:.3f}$ & ${test_report['sepsis']['precision']:.3f}$ & ${test_report['sepsis']['recall']:.3f}$ \\"

	return meta_data


if __name__ == '__main__':
	meta_data = {
		# dataset to use
		"dataset_name": "A",

		# label class to use
		"label_class": "SepsisLabel",

		# sample size of dataset
		"sample_size": 0.5,
		
		# size of test set
		"test_size": 0.3,

		# imputation method
		"imputation_method": "mean",

		# algo names
		"classifier": "svm",
		#"classifier": "decision_tree",
		"oversampler": "smote",
		"undersampler": "cluster_centroids"
	}

	# save results
	now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
	results_dir = f"results/classification/{meta_data['dataset_name']}/{now}-data_{meta_data['dataset_name']}-label_{meta_data['label_class']}-{meta_data['classifier']}-imputation_{meta_data['imputation_method']}-sample_{meta_data['sample_size']}-test_{meta_data['test_size']}"
	makedirs(results_dir, exist_ok=True)
	meta_data["results_dir"] = results_dir

	# load only A set
	print("loading data...")
	dfa = load_data_set(meta_data['dataset_name'], preprocess = True)

	# inpute missing data
	print("imputating data...")
	df = imputate_simple(dfa, meta_data['imputation_method'])

	# drop na
	df = df.dropna(axis = 1, how = "any")
	df = df.dropna(axis = 0, how = "any")

	# work on sample size
	print("sampling data...")
	df = df.sample(frac = meta_data['sample_size'])
	meta_data["samples_shape"] = df.shape
	meta_data["samples_features"] = df.columns.tolist()

	# get samples values
	X = df.drop(columns = ["SepsisLabel", "Sepsis"])
	X_feature_names = X.columns.tolist()
	meta_data["X_shape"] = X.shape
	meta_data["X_features"] = X_feature_names
	meta_data["X_description"] = json.loads(X.describe().to_json(orient="index", force_ascii=False, indent=2))
	X = X.values

	# get labels values
	y = df[meta_data['label_class']]
	meta_data["y_shape"] = y.shape
	meta_data["y_description"] = json.loads(y.astype('category').describe().to_json(orient="index", force_ascii=False, indent=2))
	y = np.array(y)

	# split in train test set
	print("splitting data...")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=meta_data['test_size'], shuffle=True)
	meta_data["x_train_shape"] = X_train.shape
	meta_data["y_train_shape"] = y_train.shape
	meta_data["X_test_shape"] = X_test.shape
	meta_data["y_test_shape"] = y_test.shape

	# scale values
	print("scaling data...")
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	joblib.dump(scaler, f"{results_dir}/scaler.joblib")

	# generate over/undersampling data late to test on "real world" data
	print("oversampling data...")
	if meta_data["oversampler"] == "smote":
		oversampler = SMOTE(n_jobs=-1)
	else:
		print("unknown oversampler:", meta_data["oversampler"])
		exit(1)
	X_train_oversampled, y_train_oversampled = oversampler.fit_resample(X_train, y_train)
	meta_data["X_train_oversampled_shape"] = X_train_oversampled.shape
	meta_data["y_train_oversampled_shape"] = y_train_oversampled.shape
	joblib.dump(oversampler, f"{results_dir}/oversampler.joblib")

	print("undersampling data...")
	if meta_data["undersampler"] == "cluster_centroids":
		undersampler = ClusterCentroids()
	else:
		print("unknown undersampler:", meta_data["undersampler"])
		exit(1)
	X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)
	meta_data["X_train_undersampled_shape"] = X_train_undersampled.shape
	meta_data["y_train_undersampled_shape"] = y_train_undersampled.shape
	joblib.dump(undersampler, f"{results_dir}/undersampler.joblib")

	# train
	meta_data["results"] = {
		**train(X_train, y_train, X_test, y_test, X_feature_names, "imbalanced", meta_data["classifier"], results_dir),
		**train(X_train_oversampled, y_train_oversampled, X_test, y_test, X_feature_names, "oversampled", meta_data["classifier"], results_dir),
		**train(X_train_undersampled, y_train_undersampled, X_test, y_test, X_feature_names, "undersampled", meta_data["classifier"], results_dir)
	}

	# save results
	with open(f"{results_dir}/results.json", "w", encoding="UTF-8") as json_file:
		json.dump(meta_data, json_file, indent=2, ensure_ascii=False)
