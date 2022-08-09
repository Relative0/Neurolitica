import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SupportFunctions import train_test_split, Bin_and_Standardize, Scoring, PrintScores

# Build the SVM model
from sklearn.svm import SVC

def Model_SVM(Questions, Bin_Length):
    from sklearn import svm
    model = SVC(kernel="linear",C=1000, gamma=.001)
    return model

def Predict_SVM(model, X, Y, TheBinsizeList):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    X_train, X_test, Y_train, Y_test = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions)
    ModelName = "SVM"
    PrintScores(Accuracy, Precision, Recall, Fscore, ModelName)
    return X_train, X_test, Y_train, Y_test

# Calculate the SHAP values using the DeepExlainer which is specific computing SHAP values with a neural network model.
def SHAPInfo_SVM(model, X_train, X_test):
    # The SHAP values
    svm_explainer = shap.KernelExplainer(model.predict, X_test)
    svm_shap_values = svm_explainer.shap_values(X_test)
    # shap.summary_plot(shap_values_Keras, X_MASQ_test_vals)  # Gives a bar graph of multiple classes
    # shap.summary_plot(shap_values_Keras[0], X_MASQ_test_vals)  # DataFrames
    return svm_explainer, svm_shap_values


def SHAPPlots_SVM(model, X_train, X_test, columns):
    explainer_SVM, shap_values_SVM = SHAPInfo_SVM(model, X_train, X_test)

    col_val = columns[0]
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = columns[:-1]

    # List = ["Summary_Single", "Summary_Bin_Bar", "Dependence", "Force_Single", "Force_All", "Waterfall"]
    List = ["Dependence"]

    for Choice in List:
        if (Choice == "Summary_Single"):
            # As there are multiple shap value configurations, one for each bin, to get the scatter plot I have to choose a
            # particular configuration configuration for bin 0, so: shap_values[0] or bin 1 shap_values[1].
            # WORKS!
            shap.summary_plot(shap_values_SVM, X_test)

        elif(Choice== "Summary_Bin_Bar"):
            # If the shap value configuration is not selected e.g. shap_values instead of shap_values[0], then all the bin
            # classes are displayed as a part of a bar graph. Otherwise, for a single shap bin configuration, if a bar graph
            # is desired, then choose plot_type="bar".
            shap.summary_plot(shap_values_SVM, X_test, plot_type="bar")  # Gives a bar graph of multiple classes

        elif (Choice== "Dependence"):
            # WORKS!
            shap.dependence_plot(col_val, shap_values_SVM, X_test_df)

        elif (Choice == "Force_Single"):
            # WORKS!
            shap.force_plot(explainer_SVM.expected_value, shap_values_SVM[10, :], X_test_df.iloc[10, :], matplotlib=True)

        elif (Choice == "Force_All"):
            plot = shap.force_plot(explainer_SVM.expected_value, shap_values_SVM, X_test, show=False)
            shap.save_html("index_SVM.htm", plot)

        elif (Choice == "Waterfall"):
            # Note, the difference in values between this and NN DeepExplainer might be because I removed a [0].
            shap.plots._waterfall.waterfall_legacy(explainer_SVM.expected_value, shap_values_SVM[0],
                                                   feature_names=X_test_df.columns)


        else:
            print("Graph Not Found.")

    # plot the SHAP values for the 10th observation





