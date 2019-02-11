import warnings
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.Model.AutoEncoder import ClassAutoEncoder
from src.Model.RestrictedBoltzmanMachine import RestrictedBoltzmanMachine



def main():
    """
    Main file for running auto-encoder
    """
    warnings.filterwarnings("ignore")

    df_numeric, df_anomaly = load_data()
    df_anomaly_array = process_data(df_anomaly)
    #auto_encoder, RBM_model, isolation_forest = estimate_models(df_anomaly_array)

    # Estimate auto-encoder
    network_structure = [15, 8]
    auto_encoder = ClassAutoEncoder(df_anomaly_array, network_structure,
                                    save_folder="/Users/az02234/Documents/Projet_AC/AnomalyDetectionAndLIME/model/",
                                    training_epochs=6)
    auto_encoder.fit()

    # Estimate restricted Boltzman machine
    RBM_model = RestrictedBoltzmanMachine(num_visible=df_anomaly_array.shape[1],
                num_hidden=15,
                visible_unit_type='gauss',
                main_dir='/Users/az02234/Documents/Projet_AC/AnomalyDetectionAndLIME/model',
                model_name='rbm_model.ckpt',
                gibbs_sampling_steps=4,
                learning_rate=0.0001,
                momentum=0.95,
                batch_size=512,
                num_epochs=100,
                verbose=1)
    #RBM_model.fit(train_set=df_anomaly_array, validation_set=None, restore_previous_model=False)

    # Estimate isolation forest`
    isolation_forest = IsolationForest(n_estimators=100, max_samples='auto', random_state=42, contamination=0.05)
    isolation_forest.fit(df_anomaly_array)



    # explainer = LimeTabularExplainer(df_anomaly_array,
    #                                  feature_names=df_anomaly.columns,
    #                                  categorical_features=[],
    #                                  categorical_names=[],
    #                                  kernel_width=3)
    #
    # np.random.seed(1)
    # i = 100
    # print(process_data(df_anomaly_array[i:i+2]))
    # exp = explainer.explain_instance(process_data(df_anomaly_array[i]), predict(process_data),
    #                                  num_features=3)
    #
    # print(exp.local_exp[1])



def predict(estimator=None, data_processor=None):
    return estimator.predict


def predict_RBM_model(df_anomaly, RBM_model):
    df_anomaly_array = process_data(df_anomaly)
    return RBM_model.predict(df_anomaly_array)


def predict_isolation_forest(df_anomaly, isolation_forest):
    df_anomaly_array = process_data(df_anomaly)
    return isolation_forest.predict(df_anomaly_array)


def load_data():
    """
    Load data
    :return df_numeric: numeric data
    :return df_anomaly:
    """
    # Put the data here
    processed_data_folder = "/Users/az02234/Documents/Projet_AC/AnomalyDetectionAndLIME/data/processed/"
    df_numeric = pd.read_csv(processed_data_folder + "df_numeric.csv")
    df_anomaly = pd.read_csv(processed_data_folder + "df_anomaly.csv")
    return df_numeric, df_anomaly


def process_data(df_anomaly):
    """
    Process data
    :param df_anomaly:
    :return df_anomaly_array: training data as numpy array
    """

    # From pandas dataframe to array
    df_anomaly_array = np.array(df_anomaly)

    # Remove invariant array
    list_invariant_array = []
    for c in range(df_anomaly_array.shape[1]):
        if df_anomaly_array[:, c].std() == 0:
            list_invariant_array.append(c)
    df_anomaly_array = np.delete(df_anomaly_array, list_invariant_array, 1)

    # Normalize data by removing mean and dividing by standard deviation
    cols_mean = []
    cols_std = []
    for c in range(df_anomaly_array.shape[1]):
        cols_mean.append(df_anomaly_array[:, c].mean())
        cols_std.append(df_anomaly_array[:, c].std())
        df_anomaly_array[:, c] = (df_anomaly_array[:, c] - cols_mean[-1]) / cols_std[-1]

    return df_anomaly_array


if __name__ == '__main__':
    main()