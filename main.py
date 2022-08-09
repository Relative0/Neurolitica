
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Import local methods
from SupportFunctions import BinConfigurations, MultiBinConfigurations, Kfold, train_test_split, Bin_and_Standardize, \
    BinData
from Create_or_Choose_Dataset import Get_or_Create_Questionnaire
from Plots import Choose_Plots, Get_Data
from Methods_NN import Compute_Subsets_NN, SHAP_Averaging, SHAPValues_NN, Train_NN, Compute_Individual_Predictions

def ChooseMethod(TrainTest_Dict, Choice, Questions, SHAP_Analysis, Bin_Length):
    X_train = TrainTest_Dict["X_train"]
    X_test = TrainTest_Dict["X_test"]
    Y_train = TrainTest_Dict["Y_train"]
    Y_test = TrainTest_Dict["Y_test"]

    if(Choice == "NN"):
        from Methods_NN import Predict_NN, SHAPPlots_NN, Model_NN, Train_NN
        # Make the model that of a NN
        model = Train_NN(X_train, Y_train, Questions, Bin_Length)
        Predict_NN(model, X_test, Y_test, Bin_arr)
        if SHAP_Analysis:
            SHAPPlots_NN(model, X_train, X_test, df_Columns)
    elif(Choice == "SVM"):
        from Methods_SVM import Predict_SVM, SHAPPlots_SVM, Model_SVM
        model = Model_SVM(Questions,Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_SVM(model, X, Y, Bin_arr)
        if SHAP_Analysis:
            SHAPPlots_SVM(model, X_train, X_test, df_Columns)
    elif (Choice == "RFR"):
        from Methods_RFR import Predict_RFR, SHAPPlots_RFR, Model_RFR
        model = Model_RFR(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_RFR(model, X, Y, Bin_arr)
        if SHAP_Analysis:
            SHAPPlots_RFR(model, X_train, X_test, df_Columns)
    elif (Choice == "EGB"):
        from Methods_EGB import Predict_EGB, SHAPPlots_EGB, Model_EGB
        model = Model_EGB(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_EGB(model, X, Y, Bin_arr)
        if SHAP_Analysis:
            SHAPPlots_EGB(model, X_train, X_test, df_Columns)
    elif (Choice == "KNN"):
        from Methods_KNN import Predict_KNN, SHAPPlots_KNN, Model_KNN
        model = Model_KNN(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_KNN(model, X, Y, Bin_arr)
        if SHAP_Analysis:
            SHAPPlots_KNN(model, X_train, X_test, df_Columns)
    elif (Choice == "H20_RF"):
        from Methods_H20RF import Predict_H20RF, SHAPPlots_H20RF, Model_H20RF
        model = Model_H20RF(Questions, Bin_Length)
        X_train, X_test, Y_train, Y_test = Predict_H20RF(model, X, Y, Bin_arr, df_Columns)
        if SHAP_Analysis:
            SHAPPlots_H20RF(model, X_train, X_test, df_Columns)
    else:
        print("Selection Not Found.")

    return model


# prepare cross validation
# kfold = KFold(n_splits=10, shuffle=False, random_state=None)
# enumerate splits

# Create or import questionnaire dataframe
Num_Questions, df = Get_or_Create_Questionnaire()
Ind_var_names = df.columns


df["Sum_of_Scores"]=df.iloc[:,:].sum(axis=1)
df_Columns = df.columns

Bin = BinConfigurations()
Bin_arr = Bin[1]
Bin_Length = len(Bin_arr) + 1

MultiBin = MultiBinConfigurations()


# -----------------------------------------------------------------------
# Prediction comparisons between the full questionnaire and the question-by-question subset.
# Compute_Subsets_NN(df, MultiBin,2)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Prediction comparisons between the full questionnaire and each of the single questions.
# Compute_Individual_Predictions(df, Bin,2)
# -----------------------------------------------------------------------


# # -----------------------------------------------------------------------
# # Uncomment if Kfold is wanted (need to re-check/work this).
# # Kfold(model, X, Y, Bin)
# # -----------------------------------------------------------------------
#
X = df.iloc[:, 0:-1]
Y = df.iloc[:, -1:]

X_arr = X.values
Y_arr = Y.values
X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=.2)
X_train, X_test, Y_train, Y_test = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, Bin_arr)
TrainTest_Dict = {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_test}


# # -----------------------------------------------------------------------
# # Uncomment if we want to find the average SHAP value.
# # SHAPConfig_Ave, SHAPFull_Ave = SHAP_Averaging(X_arr, Y_arr ,Bin, X.columns)
# # print(AveSHAP)
# # -----------------------------------------------------------------------
#
# # -----------------------------------------------------------------------
# # Uncomment if we want to Plot.

# SHAP data per questions and bin configurations for multiple bins.
FullSHAPData = Get_Data(MultiBin, df, X.columns)

# Bined value for subject data - for single bin (different then that of MultiBin).
BinnedData = BinData(Y_arr, BinConfigurations())

df["Bin"] = BinnedData
Data = dict({"BinData": df, "SHAPData": FullSHAPData})
# model = Train_NN(X_train, Y_train, Num_Questions, Bin_Length)
# # Uncomment if we want to get the SHAP single question predictions, and both total and configuration Average SHAP values.
# Need to update Choose_Plots to use the FullSHAPData as well
Choose_Plots(Data)
# # -----------------------------------------------------------------------
#
# # -----------------------------------------------------------------------
# # Uncomment one of these for particular SHAP analysis.
# DO_SHAP = True
# ChooseMethod(TrainTest_Dict,"NN", Num_Questions, DO_SHAP, Bin_Length)
# # ChooseMethod(TrainTest_Dict,"SVM",Num_Questions, DO_SHAP, Bin_Length)
# # ChooseMethod(TrainTest_Dict,"RFR",Num_Questions, DO_SHAP, Bin_Length)
# # ChooseMethod(TrainTest_Dict,"EGB",Num_Questions, DO_SHAP, Bin_Length)
# # ChooseMethod(TrainTest_Dict,"KNN",Num_Questions, DO_SHAP, Bin_Length)
#
# # H20 throws an error.
# # ChooseMethod("H20_RF")
# # -----------------------------------------------------------------------