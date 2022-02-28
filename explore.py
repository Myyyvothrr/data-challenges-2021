import pandas as pd
import numpy as np
import pysubgroup as ps
import json


def load_dataset(dataset_path: str):
	df = pd.read_csv(dataset_path,
					 sep=";",
					 dtype={
						"Gender": 'category',
						"Unit1": 'category',
						"Unit2": 'category',
						"SepsisLabel": 'category',
						"PatientID": 'category'
					}
	)

	return df


def stats(df):
	stats = {}

	# base stats
	samples_count, features_count = df.shape
	stats["samples"] = samples_count
	print("# samples", samples_count)
	stats["features"] = features_count
	print("# features", features_count)

	# numerical features
	description_numeric = df.describe()
	numerical_features_count = description_numeric.shape[1]
	stats["numerical_features"] = numerical_features_count
	print("# numerical features", numerical_features_count)
	description_numeric.T.to_csv("data/all_numerical_features.csv", sep=",", index=True, float_format='%.2f')

	# categorical features
	description_categorical = df.describe(include=['category'])
	categorical_features_count = description_categorical.shape[1]
	stats["categorical_features"] = categorical_features_count
	print("# categorical features", categorical_features_count)
	description_categorical.to_csv("data/all_categorical_features.csv", sep=",", index=True)

	# more info about gender
	df_males = df.loc[df['Gender'] == '1']
	male_patients_count = len(df_males["PatientID"].unique())
	stats["male_patients"] = male_patients_count
	print("#male patients", male_patients_count)

	# more info about sepsis
	df_sepsis = df.loc[df['SepsisLabel'] == '1']
	sepsis_patients_count = len(df_sepsis["PatientID"].unique())
	stats["sepsis_patients"] = sepsis_patients_count
	print("#sepsis patients", sepsis_patients_count)

	# male patients with sepsis
	df_males_with_sepsis = df.loc[(df['Gender'] == '1') & (df['SepsisLabel'] == '1')]
	male_sepsis_patients_count = len(df_males_with_sepsis["PatientID"].unique())
	stats["male_sepsis_patients"] = male_sepsis_patients_count
	print("#male sepsis patients", male_sepsis_patients_count)

	# female patients with sepsis
	df_females_with_sepsis = df.loc[(df['Gender'] == '0') & (df['SepsisLabel'] == '1')]
	female_sepsis_patients_count = len(df_females_with_sepsis["PatientID"].unique())
	stats["female_sepsis_patients"] = female_sepsis_patients_count
	print("#female sepsis patients", female_sepsis_patients_count)

	with open("data/all_stats.json", "w", encoding="UTF-8") as json_file:
		json.dump(stats, json_file, ensure_ascii=False, indent=2)


def subgroups(df):
	sepsis_target = '1'
	results = 10
	depth = 20

	target = ps.BinaryTarget('SepsisLabel', sepsis_target)
	searchspace = ps.create_selectors(df, ignore=['SepsisLabel', 'PatientID'])
	task = ps.SubgroupDiscoveryTask(df, target, searchspace,
									result_set_size=results,
									depth=depth,
									qf=ps.WRAccQF()
	)
	result = ps.BeamSearch().execute(task)
	result.to_dataframe().to_csv(f"data/all_subgroups_sepsis{sepsis_target}_depth{depth}_results{results}.csv", sep=",", index=True)


def timeseries(df):
	# get length of time series per patient:
	# ICULOS ICU length-of-stay (hours since ICU admit)

	# get max ICULOS per patient -> length of time series
	patients_iculos_max = df.groupby(by=['PatientID'])['ICULOS'].max()
	patients_iculos_max.describe().T.to_csv("data/all_timeseries.csv", sep=",", index=True)


if __name__ == "__main__":
	df = load_dataset("data/train_all_merged.csv")

	stats(df)

	#subgroups(df)

	timeseries(df)

	print("done")
