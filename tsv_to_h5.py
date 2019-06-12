import pandas as pd

features = "feature_data/feature_values.tsv"
job_desc = "feature_data/job_descriptions.tsv"
df_features = pd.read_csv(features, sep="\t")
df_job_desc = pd.read_csv(job_desc, sep="\t")

df_features.to_hdf('feature_data/feature_values.h5', key='f', mode='w')
df_job_desc.to_hdf('feature_data/job_descriptions.h5', key='d', mode='w')
