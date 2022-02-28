import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
from os import listdir, path


def load_dataset(dataset_paths: List[str]):
	# get all dataset files
	dataset_files = []
	for dataset_path in dataset_paths:
		dataset_files += [
			(f, dataset_path) for f
			in listdir(dataset_path)
			if path.isfile(path.join(dataset_path, f))
		]

	# load into dataframe
	dfs = []
	for (f, p) in tqdm(dataset_files, desc="loading data"):
		df = pd.read_csv(path.join(p, f), sep="|")

		# add patient id (remove .psv)
		df["PatientID"] = f[:-4]

		dfs.append(df)

	# concat all
	df = pd.concat(dfs, ignore_index=True, sort=False)

	# update data types
	df["Gender"] = df["Gender"].astype('category')
	df["Unit1"] = df["Unit1"].astype('category')
	df["Unit2"] = df["Unit2"].astype('category')
	df["SepsisLabel"] = df["SepsisLabel"].astype('category')
	df["PatientID"] = df["PatientID"].astype('category')

	# cleanup
	del dataset_files
	del dfs

	return df


if __name__ == "__main__":
	df = load_dataset(["data/training", "data/training_setB"])

	# save as one file
	df.to_csv("data/all.csv", sep=";", index=False)
