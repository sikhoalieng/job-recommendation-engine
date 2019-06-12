from keras.models import load_model
import tensorflow as tf
from utils import *
import numpy as np
import pandas as pd
import flask


# initialize our Flask application
app = flask.Flask(__name__)


def load_data(features, job_desc, glove_file, model_file):
    global df_features, df_job_desc, word_to_index, degree_type_full, managed_others_full, years_exp_full, model, graph
    df_features = pd.read_hdf(features, 'f')
    df_job_desc = pd.read_hdf(job_desc, 'd')
    word_to_index = get_words_to_index(glove_file)
    degree_type_full = df_features["DegreeType"].unique()
    managed_others_full = df_features["ManagedOthers"].unique()
    years_exp_full = df_features["YearsOfExperience"].unique()

    # Returns a compiled model
    model = load_model(model_file)
    graph = tf.get_default_graph()


def preprocess(major_input, degree_type_input, managed_others_input, years_exp_input):
    df_job_titles = df_features[["JobID", "Title"]].drop_duplicates(subset="Title")
    job_titles_short = np.zeros((df_job_titles.shape[0],), dtype=object)
    # Reduce noise in job titles
    for index, value in enumerate(df_job_titles["Title"]):
        job_titles_short[index] = value[:35]
    global df_jobs_unique
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

    return major, title, degree_type, managed_others, years_exp


@app.route("/predict", methods=["POST"])
def predict():
    post = flask.request.get_json()

    major_input = post['major_input']
    degree_type_input = post['degree_type_input']
    managed_others_input = post['managed_others_input']
    years_exp_input = post['years_exp_input']
    k = post['k']

    major, title, degree_type, managed_others, years_exp = \
        preprocess(major_input, degree_type_input, managed_others_input, years_exp_input)

    with graph.as_default():
        # Get predictions
        pred = model.predict([major, title, degree_type, managed_others, years_exp])
        # Get top k predictions (recommendations)
        top_k = sorted(range(len(pred)), key=lambda i: pred[i], reverse=True)[:k]
        recommendations = df_jobs_unique.iloc[top_k].reset_index(drop=True)
        recommendations["Score"] = pred[top_k]

        return recommendations.to_json(orient='index')


if __name__ == "__main__":
    features = "feature_data/feature_values.h5"
    job_desc = "feature_data/job_descriptions.h5"
    model_file = 'model/neural_network.h5'
    glove_file = 'feature_data/GloVe/glove.6B.50d.txt'

    # Load data to memory
    load_data(features, job_desc, glove_file, model_file)

    # Start server
    app.run()
