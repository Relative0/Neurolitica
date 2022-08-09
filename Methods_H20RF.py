import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
h2o.init()



import shap
import numpy as np
import pandas as pd
from sklearn import ensemble
import matplotlib.pyplot as plt
from SupportFunctions import train_test_split, Bin_and_Standardize, Scoring, PrintScores, Round
from sklearn import neighbors
def Model_H20RF(Questions, Bin_Length):
    model = H2ORandomForestEstimator(ntrees=200, max_depth=20, nfolds=10)
    return model

def Predict_H20RF(model, X, Y, TheBinsizeList, dataframe_cols):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    X_train, X_test, Y_train, Y_test = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList)
    X_train_hex = h2o.H2OFrame(X_train)
    X_test_hex = h2o.H2OFrame(X_test)
    dataframe_cols_X = dataframe_cols[0:-1]
    dataframe_cols_Y = dataframe_cols[-1]
    model.train(x=dataframe_cols_X, y=dataframe_cols_Y, training_frame=X_train_hex)

    # model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    predictions_rounded = Round(predictions, 0)
    Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions_rounded)
    ModelName = "H20RF"
    PrintScores(Accuracy, Precision, Recall, Fscore, ModelName)

    return X_train, X_test, Y_train, Y_test

class H2OProbWrapper:
    def __init__(self, h2o_model, feature_names):
        self.h2o_model = h2o_model
        self.feature_names = feature_names
def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        self.dataframe= pd.DataFrame(X, columns=self.feature_names)
        self.predictions = self.h2o_model.predict(h2o.H2OFrame(self.dataframe)).as_data_frame().values
        return self.predictions.astype('float64')[:,-1]

def SHAPInfo_H20RF(model, X_train, X_test):
    # The SHAP values
    h2o_wrapper = H2OProbWrapper(model, X_train.columns)
    h2o_rf_explainer = shap.KernelExplainer(h2o_wrapper.predict_binary_prob, X_test)
    h2o_rf_shap_values = h2o_rf_explainer.shap_values(X_test)

    return h2o_rf_explainer, h2o_rf_shap_values


def SHAPPlots_H20RF(model, X_train, X_test, columns):
    explainer_H20RF, shap_values_H20RF = SHAPInfo_H20RF(model, X_train, X_test)

    col_val = columns[0]
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = columns[:-1]

    # List = ["Summary_Single", "Summary_Bin_Bar", "Dependence", "Force_Single", "Force_All", "Waterfall"]
    List = ["Waterfall"]

    for Choice in List:
        if (Choice == "Summary_Single"):
            # As there are multiple shap value configurations, one for each bin, to get the scatter plot I have to choose a
            # particular configuration configuration for bin 0, so: shap_values[0] or bin 1 shap_values[1].
            shap.summary_plot(shap_values_H20RF, X_test)

        elif (Choice == "Summary_Bin_Bar"):
            # If the shap value configuration is not selected e.g. shap_values instead of shap_values[0], then all the bin
            # classes are displayed as a part of a bar graph. Otherwise, for a single shap bin configuration, if a bar graph
            # is desired, then choose plot_type="bar".
            shap.summary_plot(shap_values_H20RF, X_test, plot_type="bar")  # Gives a bar graph of multiple classes

        elif (Choice == "Dependence"):
            #
            shap.dependence_plot(col_val, shap_values_H20RF, X_test_df)

        elif (Choice == "Force_Single"):
            # plot the SHAP values for the 10th observation
            shap.force_plot(explainer_H20RF.expected_value, shap_values_H20RF[10, :], X_test_df.iloc[10, :],
                            matplotlib=True)

        elif (Choice == "Force_All"):
            plot = shap.force_plot(explainer_H20RF.expected_value, shap_values_H20RF, X_test, show=False)
            shap.save_html("index_H20RF.htm", plot)

        elif (Choice == "Waterfall"):
            # Note, the difference in values between this and NN DeepExplainer might be because I removed a [0].
            shap.plots._waterfall.waterfall_legacy(explainer_H20RF.expected_value, shap_values_H20RF[0],
                                                   feature_names=X_test_df.columns)

        else:
            print("Plot not found.")


