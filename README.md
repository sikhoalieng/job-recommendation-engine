# Career Oracle - Job Recommendation Engine

This is the neural network recommendation engine that powers our online career exploration tool: [Career Oracle](http://teamd5.s3-website-us-east-1.amazonaws.com)

![Demo](https://github.com/sikhoalieng/job-recommendation-engine/blob/master/img/CareerOracle_demo.gif)

## Getting Started

The following folder structure is recommended

```
Job Recommendation Engine
│   predict.py
│   predict_api.py
│   README.md
│   requirements.txt
│   sample_request.py
|   test.py
│   utils.py
│
└───feature_data
│   │   feature_values.h5
│   │   job_descriptions.h5
│   │
│   └───GloVe
│       │   glove.6B.50d.txt
│   
└───model
│   │   neural_network.h5
│   
└───recommendations  # optional
    │   Computer Science_Master's_2_Yes.csv
    │   Computer Science_Master's_2_Yes.json
```

### Prerequisites

The following Python 3.5 packages are required; versions are recommended

```
tensorflow~=1.10.0
scikit-learn~=0.20.0
numpy~=1.15.2
pandas~=0.23.4
h5py~=2.8.0
pytables~=3.4.4
keras~=2.2.2
flask~=0.12
```

### Installing

Install required packages individually

```
pip install numpy~=1.15.2
```

Or pip install using requirements.txt file

```
pip install -r requirements.txt
```

End with an example of getting some data out of the system or using it for a little demo

## Querying the model

Queries below will return a pandas DataFrame

#### Top k recommendations
User profile (example)
* Major: Computer Science
    * Cannot be empty string (if major is not applicable, enter relevant industry)
* Degree Type: Master's
    * Possible choices (case-sensitive): ["Bachelor's", "Master's", "Associate's", 'None', 'High School', 'Vocational', 'PhD']
* Managed Others: Yes
    * Possible choices (case-sensitive): ['Yes', 'No']
* Years of Experience: 2

```python
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

# Print results
print(recommendations)  # Pandas DataFrame
# print(recommendations.to_json(orient='index'))  # JSON format
```

#### Output

Printing the recommendations (top 100):

```
    Title                                               Description                                         Score
0   Web Developer - Java, Hibernate, Spring, REST ...   Web Developer - Java, Hibernate, Spring, REST ...   0.998005 
1                 ABAP Sr. Developer - Sr. Programmer   \rABAP Sr. Developer - Sr. Programmer - Inform...   0.993852
2                           Sr. Oracle Database Admin   Our Major Financial Client is looking for a…\r...   0.978664
.                           .                                                   .                               .
.                           .                                                   .                               .
.                           .                                                   .                               .
99           NOC Help Desk Technician - Cisco Routers   \rNOC Center Technician\r  \rA national IT Tel...   0.734699
```

## References

* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) - Word embedding used
* [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf) - Negative sampling method used

## Authors

* **Si Khoa Lieng** - Job Recommendation Engine

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

Big thanks to members of team D5 for their contributions to the Career Oracle project:
* John Kim
* James Ching
* Gil Ferreira
* Jianglu Zhang
