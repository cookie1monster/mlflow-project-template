from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

import mlflow
import mlflow.xgboost
import click
import pandas as pd


@click.command(help="Train xgboost model")
@click.option("--training_data", default="./data/wine-quality.csv")
@click.option("--colsample_bytree", type=click.FLOAT, default=1.0)
@click.option("--subsample", type=click.FLOAT, default=1.0)
@click.option("--target-name", type=click.STRING, default="quality")
def run(training_data, colsample_bytree, subsample, target_name):
    df = pd.read_csv(training_data)
    X = df[set(df.columns) - set(target_name)]
    y = df[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    mlflow.xgboost.autolog()

    with mlflow.start_run():
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'colsample_bytree': colsample_bytree,
            'subsample': subsample,
            'seed': 42,
        }
        model = xgb.train(params, dtrain, evals=[(dtrain, 'train')])

        y_pred = model.predict(dtest)

        val_mse = mean_squared_error(y_test, y_pred)
        val_mae = mean_absolute_error(y_test, y_pred)

        print(f'mse: {val_mse}, mse: {val_mae}')

        mlflow.log_metrics({'val_mse': val_mse, 'val_mae': val_mae})


if __name__ == '__main__':
    run()
