### Run mlflow server 
`mlflow server`

### ETL DATA
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow run . --entry-point etl_data --param-list 'training_data'='./data/wine-quality.csv' --no-conda --experiment-name expr-name3`

### Grid search
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow run . --entry-point grid_search --param-list 'max_p'=4 --param-list 'max_runs'=6 --param-list 'training_data'='./data/wine-quality.csv' --no-conda --experiment-name expr-name3`

### Train
`MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow run . --entry-point train --param-list 'subsample'=.4 --param-list 'training_data'='./data/wine-quality.csv' --no-conda --experiment-name expr-name3`

