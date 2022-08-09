import shap
import numpy as np
import pandas as pd
from sklearn import ensemble
import matplotlib.pyplot as plt
from SupportFunctions import train_test_split, Bin_and_Standardize, Scoring, PrintScores, Round
def Model_EGB(Questions, Bin_Length):
    n_estimators = 500
    model = ensemble.GradientBoostingClassifier(
                n_estimators=n_estimators,
                validation_fraction=0.2,
                n_iter_no_change=5,
                tol=0.01,
                random_state=0)
    return model


def Predict_EGB(model, X, Y, TheBinsizeList):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    X_train, X_test, Y_train, Y_test = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList)

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    predictions_rounded = Round(predictions, 0)
    Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions_rounded)
    ModelName = "EGB"
    PrintScores(Accuracy, Precision, Recall, Fscore, ModelName)

    return X_train, X_test, Y_train, Y_test


# Calculate the SHAP values using the DeepExlainer which is specific computing SHAP values with a neural network model.
def SHAPInfo_EGB(model, X_train, X_test):
    # The SHAP values
    egb_explainer = shap.KernelExplainer(model.predict, X_test)
    egb_shap_values = egb_explainer.shap_values(X_test)

    return egb_explainer, egb_shap_values


def SHAPPlots_EGB(model, X_train, X_test, columns):
    explainer_EGB, shap_values_EGB = SHAPInfo_EGB(model, X_train, X_test)

    col_val = columns[0]
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = columns[:-1]

    # List = ["Summary_Single", "Summary_Bin_Bar", "Dependence", "Force_Single", "Force_All", "Waterfall"]
    List = [ "Waterfall"]

    for Choice in List:
        if (Choice == "Summary_Single"):
            # As there are multiple shap value configurations, one for each bin, to get the scatter plot I have to choose a
            # particular configuration configuration for bin 0, so: shap_values[0] or bin 1 shap_values[1].
            shap.summary_plot(shap_values_EGB, X_test)

        elif (Choice == "Summary_Bin_Bar"):
            # If the shap value configuration is not selected e.g. shap_values instead of shap_values[0], then all the bin
            # classes are displayed as a part of a bar graph. Otherwise, for a single shap bin configuration, if a bar graph
            # is desired, then choose plot_type="bar".
            shap.summary_plot(shap_values_EGB, X_test, plot_type="bar")  # Gives a bar graph of multiple classes

        elif (Choice == "Dependence"):
            #
            shap.dependence_plot(col_val, shap_values_EGB, X_test_df)

        elif (Choice == "Force_Single"):
            # plot the SHAP values for the 10th observation
            shap.force_plot(explainer_EGB.expected_value, shap_values_EGB[10, :], X_test_df.iloc[10, :],
                            matplotlib=True)

        elif (Choice == "Force_All"):
            plot = shap.force_plot(explainer_EGB.expected_value, shap_values_EGB, X_test, show=False)
            shap.save_html("index_EGB.htm", plot)

        elif (Choice == "Waterfall"):
            # Note, the difference in values between this and NN DeepExplainer might be because I removed a [0].
            shap.plots._waterfall.waterfall_legacy(explainer_EGB.expected_value, shap_values_EGB[0],
                                                   feature_names=X_test_df.columns)

        else:
            print("Plot not found.")

