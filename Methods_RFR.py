import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from SupportFunctions import train_test_split, Bin_and_Standardize, Scoring, PrintScores, Round

# Build the RFR model
from sklearn.svm import SVC

def Model_RFR(Questions, Bin_Length):
    model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    return model


def Predict_RFR(model, X, Y, TheBinsizeList):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    X_train, X_test, Y_train, Y_test = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList)

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    predictions_rounded = Round(predictions, 0)
    Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions_rounded)
    ModelName = "RFR"
    PrintScores(Accuracy, Precision, Recall, Fscore, ModelName)

    # print(model.feature_importances_)
    # importances = model.feature_importances_
    # indices = np.argsort(importances)
    # features = X_train.columns
    #
    # Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions)
    # ModelName = "RFR"
    # PrintScores(Accuracy, Precision, Recall, Fscore, ModelName)
    return X_train, X_test, Y_train, Y_test

# Calculate the SHAP values using the DeepExlainer which is specific computing SHAP values with a neural network model.
def SHAPInfo_RFR(model, X_train, X_test):
    # The SHAP values
    rfr_explainer = shap.KernelExplainer(model.predict, X_test)
    rfr_shap_values = rfr_explainer.shap_values(X_test)
   
    return rfr_explainer, rfr_shap_values


def SHAPPlots_RFR(model, X_train, X_test, columns):
    explainer_RFR, shap_values_RFR = SHAPInfo_RFR(model, X_train, X_test)

    col_val = columns[0]
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = columns[:-1]

    # List = ["Summary_Single", "Summary_Bin_Bar", "Dependence", "Force_Single", "Force_All", "Waterfall"]
    List = ["Waterfall"]

    for Choice in List:
        if (Choice == "Summary_Single"):
            # As there are multiple shap value configurations, one for each bin, to get the scatter plot I have to choose a
            # particular configuration configuration for bin 0, so: shap_values[0] or bin 1 shap_values[1].
            # WORKS!
            shap.summary_plot(shap_values_RFR, X_test)

        elif(Choice== "Summary_Bin_Bar"):
            # If the shap value configuration is not selected e.g. shap_values instead of shap_values[0], then all the bin
            # classes are displayed as a part of a bar graph. Otherwise, for a single shap bin configuration, if a bar graph
            # is desired, then choose plot_type="bar".
            shap.summary_plot(shap_values_RFR, X_test, plot_type="bar")  # Gives a bar graph of multiple classes

        elif (Choice== "Dependence"):
            # WORKS!
            shap.dependence_plot(col_val, shap_values_RFR, X_test_df)

        elif (Choice == "Force_Single"):
            # WORKS!
            shap.force_plot(explainer_RFR.expected_value, shap_values_RFR[10, :], X_test_df.iloc[10, :], matplotlib=True)

        elif (Choice == "Force_All"):
            plot = shap.force_plot(explainer_RFR.expected_value, shap_values_RFR, X_test, show=False)
            shap.save_html("index_RFR.htm", plot)

        elif (Choice == "Waterfall"):
            # Note, the difference in values between this and NN DeepExplainer might be because I removed a [0].
            shap.plots._waterfall.waterfall_legacy(explainer_RFR.expected_value, shap_values_RFR[0],
                                                   feature_names=X_test_df.columns)

        else:
            print("Graph Not Found.")
    # Not working..
    # shap.plots.waterfall(shap_values[0])
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
    # shap.force_plot(explainer.expected_value, shap_values[0, :], X_train)



    # plot the SHAP values for the 10th observation





