# imports
from locale import D_FMT
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

from ml_flow_test import EXPERIMENT_NAME

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[UK] [London] [pannich] TexiFareModel + 1.0"
# 🚨 replace with your country code, city, github_nickname and model name and version

class Trainer():

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe #no need to return anything just store pipeline in the background


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline() #running the function set_pipeline() and sotre self.pipeline
        #self.mlflow_log_param
        '''returns a trained pipelined model'''
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test) #self.pipeline is your pipeline model
        rmse = compute_rmse(y_pred, y_test)
        return rmse


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # train
    trainer = Trainer(X_train,y_train)
    # can chagne experiment name

    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_val,y_val)
    print(rmse)
    #client = trainer.mlflow_client()
    trainer.mlflow_log_metric("rmse", 4.5)
    trainer.mlflow_log_param("model", "linear")
    trainer.mlflow_log_metric("rmse", 5.2)
    trainer.mlflow_log_param("model", "Randomforest")


# for model in ["linear", "Randomforest"]:
#     run = client.create_run(experiment_id)
#     client.log_metric(run.info.run_id, "rmse", 4.5)
#     client.log_param(run.info.run_id, "model", model)
#     client.log_param(run.info.run_id, "student_name", yourname)

# self.mlflow_log_param(param_name, param_value)
# self.mlflow_log_metric(metric_name, metric_value)
