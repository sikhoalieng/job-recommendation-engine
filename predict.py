from keras.models import load_model
from utils import *
import numpy as np
import pandas as pd
import time
pd.options.display.max_columns = 20
pd.options.display.max_rows = 100


def load_data(features, job_desc, glove_file, model_file):
    df_features = pd.read_hdf(features, 'f')
    df_job_desc = pd.read_hdf(job_desc, 'd')
    word_to_index = get_words_to_index(glove_file)
    degree_type_full = df_features["DegreeType"].unique()
    managed_others_full = df_features["ManagedOthers"].unique()
    years_exp_full = df_features["YearsOfExperience"].unique()

    # Returns a compiled model
    model = load_model(model_file)

    return df_features, df_job_desc, word_to_index, degree_type_full, managed_others_full, years_exp_full, model


def preprocess(df_features, df_job_desc, word_to_index, degree_type_full, managed_others_full, years_exp_full,
               major_input, degree_type_input, managed_others_input, years_exp_input):
    df_job_titles = df_features[["JobID", "Title"]].drop_duplicates(subset="Title")
    job_titles_short = np.zeros((df_job_titles.shape[0],), dtype=object)
    # Reduce noise in job titles
    for index, value in enumerate(df_job_titles["Title"]):
        job_titles_short[index] = value[:35]
    df_jobs_unique = \
        df_job_titles["JobID"].to_frame().merge(df_job_desc, on="JobID", how="inner")[["Title", "Description"]]

    # Make user inputs the same size as number of jobs we enter into the model (tensorflow requirement)
    major = [major_input for i in range(df_jobs_unique.shape[0])]
    degree_type = [degree_type_input for i in range(df_jobs_unique.shape[0])]
    managed_others = [managed_others_input for i in range(df_jobs_unique.shape[0])]
    years_exp = [years_exp_input for i in range(df_jobs_unique.shape[0])]

    # Preprocess data for neural network
    major = string_to_indices(np.array(major), word_to_index, 30)
    title = string_to_indices(job_titles_short, word_to_index, 30)
    degree_type = strings_to_one_hot(degree_type_full, np.array(degree_type))
    managed_others = strings_to_one_hot(managed_others_full, np.array(managed_others))
    years_exp = scale_values(years_exp_full, np.array(years_exp))

    return major, title, degree_type, managed_others, years_exp, df_jobs_unique


def predict(major, title, degree_type, managed_others, years_exp, df_jobs_unique, model, k=50):
    pred = model.predict([major, title, degree_type, managed_others, years_exp])

    # Get top k predictions (recommendations)
    top_k = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k]

    recommendations = df_jobs_unique.iloc[top_k].reset_index(drop=True)
    recommendations["Score"] = pred[top_k]

    return recommendations


if __name__ == "__main__":
    start_time = time.time()

    features = "feature_data/feature_values.h5"
    job_desc = "feature_data/job_descriptions.h5"
    model_file = 'model/neural_network.h5'
    glove_file = 'feature_data/GloVe/glove.6B.50d.txt'

    major_input = "Accounting"
    degree_type_input = "Master's"
    managed_others_input = "Yes"
    years_exp_input = 2
    k = 100

    # Load data to memory
    df_features, df_job_desc, word_to_index, degree_type_full, managed_others_full, years_exp_full, model = \
        load_data(features, job_desc, glove_file, model_file)
    load_time = time.time() - start_time
    print(load_time, "seconds to load data to memory")

    # Pre-process inputs
    major, title, degree_type, managed_others, years_exp, df_jobs_unique = \
        preprocess(df_features, df_job_desc, word_to_index, degree_type_full, managed_others_full, years_exp_full,
                   major_input, degree_type_input, managed_others_input, years_exp_input)

    # Get predictions
    recommendations = predict(major, title, degree_type, managed_others, years_exp, df_jobs_unique, model, k)

    elapsed_time = time.time() - start_time
    print(elapsed_time, "seconds for data load + recommendations")
    print(recommendations)

    recommendations.to_csv("recommendations/" + major_input + "_"
                           + degree_type_input + "_"
                           + str(years_exp_input) + "_"
                           + managed_others_input + ".csv", index=False)
