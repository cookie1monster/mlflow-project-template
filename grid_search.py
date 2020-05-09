from concurrent.futures import ThreadPoolExecutor

import click
import itertools
import yaml
import random
import numpy as np

import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.projects
from mlflow.tracking.client import MlflowClient

_inf = np.finfo(np.float64).max


@click.command(help="Perform grid search over train")
@click.option("--training_data", type=click.STRING, default="./data/wine-quality.csv")
@click.option("--config_path", type=click.STRING, default="./data/config.yml")
@click.option("--max-runs", type=click.INT, default=32,
              help="Maximum number of runs to evaluate.")
@click.option("--max-p", type=click.INT, default=2,
              help="Maximum number of parallel runs.")
@click.option("--metric", type=click.STRING, default="mse",
              help="Metric to optimize on.")
@click.option("--seed", type=click.INT, default=97531,
              help="Seed for the random generator")
def run(training_data, config_path, max_runs, max_p, metric, seed):
    val_metric = f"val_{metric}"

    np.random.seed(seed)
    tracking_client = mlflow.tracking.MlflowClient()

    def new_eval(experiment_id):
        def eval(parms):
            with mlflow.start_run(nested=True) as child_run:
                p = mlflow.projects.run(
                    run_id=child_run.info.run_id,
                    uri=".",
                    entry_point="train",
                    parameters={
                        "training_data": training_data,
                        "colsample_bytree": str(parms['colsample_bytree']),
                        "subsample": str(parms['subsample']),
                        "target-name": str(parms['target_name'])},
                    experiment_id=experiment_id,
                    synchronous=True, use_conda=False)
                succeeded = p.wait()
            if succeeded:
                training_run = tracking_client.get_run(p.run_id)
                metrics = training_run.data.metrics
                val_loss = metrics[val_metric]
            else:
                tracking_client.set_terminated(p.run_id, "FAILED")
                val_loss = _inf
            mlflow.log_metrics({
                val_metric: val_loss,
            })
            return p.run_id, val_loss

        return eval

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id

        with ThreadPoolExecutor(max_workers=max_p) as executor:
            _ = executor.map(new_eval(experiment_id), generate_configs(config_path, max_runs))

        # find the best run, log its metrics as the final metrics of this run.
        client = MlflowClient()
        runs = client.search_runs([experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}' ")

        print(runs)

        best_val_valid = _inf
        best_run = None
        for r in runs:
            if r.data.metrics[val_metric] < best_val_valid:
                best_run = r
                best_val_valid = r.data.metrics[val_metric]
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics({
            "val_{}".format(metric): best_val_valid,
        })


def generate_configs(path, max_number):
    with open(path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        config_space = []
        for param in params.keys():
            config_space.append(params[param])
        config_space = list(itertools.product(*config_space))

        configs = []
        for config in config_space:
            configs.append({key: value for key, value in zip(params.keys(), config)})
        random.shuffle(configs)
        return configs[:max_number]


if __name__ == '__main__':
    run()
