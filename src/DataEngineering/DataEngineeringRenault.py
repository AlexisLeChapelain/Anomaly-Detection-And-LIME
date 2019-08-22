import sys

import pandas as pd
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")


def process_data():
    """
    Process data to do simple data engineering : remove empty columns, useless keys,
    :return: None
    """

    sys.path.append("/Users/az02234/Documents/Projet_AC/AnomalyDetectionAndLIME/")

    raw_data_path = "/Users/az02234/Documents/Projet_AC/AnomalyDetectionAndLIME/data/raw/"
    processed_data_path = "/Users/az02234/Documents/Projet_AC/AnomalyDetectionAndLIME/data/processed/"

    df = pd.read_excel(raw_data_path + "C-Extraction_SMACC_MOP_FC_avec_marques 2017 Confidentiel B.xls")

    # remove empty columns
    df.drop(["Contribution commerciale pays Pièces_signé",
             "Contribution commerciale pays Accessoires_signé",
             "CA Pièces_signé",
             "CA Accessoires_signé",
             "CDV Pièces_signé",
             "CDV Accessoires_signé",
             "MCx fixes pays Pièces_signé",
             "MCx fixes pays Accessoires_signé",
             "MCx variables pays Pièces_signé",
             "MCx variables pays Accessoires_signé",
             "MCx fixes région Pièces_signé",
             "MCx fixes région Accessoires_signé",
             "Actualisation et raccord Pièces_signé",
             "Actualisation et raccord Accessoires_signé",
             "MCx fixes centraux Pièces_signé",
             "MCx fixes centraux Accessoires_signé",
             "CA R&D hors reprises de PCA_signé",
             "MOP Prestation de services_signé",
             "MAC Frais généraux Services_signé",
             "Raccord Services_signé",
             "Autres frais commerciaux centraux - Commerce_signé",
             "KPI B3 Monde_signé",
             "EV Business Margin_signé",
             "Cash Payment_signé.1",
             "Cash Payment_signé",
             "CA Avtovaz_signé",
             "Retraitement PSS P&A_signé",
             "R&D non affectée - Sites centraux_signé",
             "Passage MAC à MOP RCI part RSI_signé"], axis=1, inplace=True)

    # remove useless keys
    df.drop(columns=["Année"], inplace=True)

    # Recode categorical features with numerical values
    categorical_names = {}
    for feature in ["Marché", "Produit", "Numéro Mois Période"]:
        le = preprocessing.LabelEncoder()
        le.fit(df.loc[:, feature])
        df.loc[:, feature] = le.transform(df.loc[:, feature])
        categorical_names[feature] = le.classes_

    # Create a dataset with only numerical values, and without missing values
    print("There is {} columns and {} observations in the df dataset".format(df.shape[1], df.shape[0]))
    numerics = ["int64", "float64"]
    df_numeric = df.select_dtypes(include=numerics)
    df_numeric.dropna(inplace=True, axis=1)
    print("There is {} columns and {} observations in the df_num dataset".format(df_numeric.shape[1],
                                                                                 df_numeric.shape[0]))

    # Create a dataframe for the isolation forest by removing the keys "Marché", "Produit", "Numéro Mois Période"
    df_anomaly = df_numeric.drop(columns=["Marché", "Produit", "Numéro Mois Période"])

    df_anomaly.to_csv(processed_data_path + "df_anomaly.csv")
    df_numeric.to_csv(processed_data_path + "df_numeric.csv")


if __name__ == '__main__':
    process_data()
