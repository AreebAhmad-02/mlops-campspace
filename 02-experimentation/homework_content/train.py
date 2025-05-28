import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


# def run_train(data_path: str):

#     X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
#     X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

#     rf = RandomForestRegressor(max_depth=10, random_state=0)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_val)

#     rmse = root_mean_squared_error(y_val, y_pred)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def mlflow_run_train(data_path: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("nyc-taxi-experiment")  # optional, but good practice
    # mlflow.sklearn.autolog()  # <- Enable autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        mlflow.set_tag("developer", "areeb")

        # mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
        # mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")

        # alpha = 0.1
        # mlflow.log_param("alpha", alpha)
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        mlflow.sklearn.log_model(rf, artifact_path="models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

        # mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")


if __name__ == '__main__':
    mlflow_run_train()
