import mlflow
import click
import pandas as pd


@click.command(help="Get and transform raw data")
@click.option("--training_data", default="./data/wine-quality.csv")
def etl_data(training_data):
    with mlflow.start_run() as mlrun:
        df = pd.read_csv(training_data)
        # do some transformation
        # save data
        print("Uploading data: %s" % df.iloc[:10].to_string)
        mlflow.log_artifact(training_data, "data")


if __name__ == '__main__':
    etl_data()
