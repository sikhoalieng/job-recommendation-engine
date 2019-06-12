from predict_api import load_data, preprocess, predict

# File locations
features = "feature_data/feature_values.h5"
job_desc = "feature_data/job_descriptions.h5"
glove_file = "feature_data/GloVe/glove.6B.50d.txt"
model_file = "model/neural_network.h5"

# User profile
major_input = "Computer Science"
degree_type_input = "Master's"
managed_others_input = "Yes"
years_exp_input = 2
k = 100

# Load data to memory
df_features, df_job_desc, word_to_index, degree_type_full, managed_others_full, years_exp_full, model = \
    load_data(features, job_desc, glove_file, model_file)

# Preprocess inputs
major, title, degree_type, managed_others, years_exp, df_jobs_unique = \
    preprocess(df_features, df_job_desc, word_to_index, degree_type_full, managed_others_full, years_exp_full,
                       major_input, degree_type_input, managed_others_input, years_exp_input)

# Get predictions
recommendations = predict(major, title, degree_type, managed_others, years_exp, df_jobs_unique, model, k)
print(recommendations)
# print(recommendations.to_json(orient='index'))
