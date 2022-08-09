# Import libraries.
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from statistics import pstdev
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
from sklearn.model_selection import KFold

from DimensionalityBinning import DimensionalBinChoice
from sklearn.model_selection import train_test_split

# Average an input list.
def AverageList(InList):
    output = [sum(elem) / len(elem) for elem in zip(*InList)]
    # print(output)
    return output

def StdDevList(InList):
    output = [pstdev(elem) for elem in zip(*InList)]
    return output

def ListAverage(InList):
    output = sum(InList) / len(InList)
    # print(output)
    return output

# Standardize the dataframe.
def Standardize(data):
    if isinstance(data,np.ndarray):
        Dataframe = preprocessing.StandardScaler().fit_transform(data)
    if isinstance(data,pd.DataFrame):
        # Get column names first
        Dataframe = data.copy()
        col_names = Dataframe.columns
        features = Dataframe[col_names]
        Standardized = preprocessing.StandardScaler()
        scaler = Standardized.fit(features.values)
        features = scaler.transform(features.values)
        Dataframe[col_names] = features

    return Dataframe

# Choose whether or not to imput missing values from a dataframe.
def ChoiceImputation(df,choice):
    # We create a limit between 1 and 5 inclusive to find and remove outliers.
    limit = [1, 2, 3, 4, 5]
    df_Cleaned = df[df.isin(limit)]
    if choice == 0:
        #Don't impute
        df_Cleaned = df_Cleaned.dropna(axis = 0, how ='any')

    elif choice == 1:
        #Impute
        df_Cleaned.fillna(df_Cleaned.mean(), inplace=True)
    else:
        print("Issue in ChoiceImputation function")
    return df_Cleaned

def ChoiceImputation(df,choice):
    # We create a limit between 1 and 5 inclusive to find and remove outliers.
    limit = [1, 2, 3, 4, 5]
    df_Cleaned = df[df.isin(limit)]
    if choice == 0:
        #Don't impute
        df_Cleaned = df_Cleaned.dropna(axis = 0, how ='any')

    elif choice == 1:
        #Impute
        df_Cleaned.fillna(df_Cleaned.mean(), inplace=True)
    else:
        print("Issue in ChoiceImputation function")
    return df_Cleaned

# Round an individual value.
def Round(value, Significance):
    value = np.round(value, Significance)
    return value

# Round each value in a list.
def RoundandPercent(InList):
    rounded = ["{:.3f}".format(round(num, 3)) for num in InList]
    return rounded

def BinConfigurations():
    # bin = ['2-Bin', [0]]
    bin = ['3-Bin', [-.431, .431]]
    # bin= ['4-Bin', [-.674, 0, .674]]
    # bin = ['5-Bin', [-.842, -0.253, 0.253, .842]]
    # bin = ['6-Bin', [-0.967, -0.431, 0, 0.431, 0.967]]
    # bin = ['7-Bin', [-1.068, -0.566, -0.18, 0.18, 0.566, 1.068]]
    # bin = ['8-Bin', [-1.15, -.674, -.319, 0, .319, .674, 1.15]]
    return bin

def MultiBinConfigurations():
    # Defines the number and AUC of each bin array.
    TwoBin = ['2-Bin',[0]]
    ThreeBin = ['3-Bin',[-.431, .431]]
    FourBin = ['4-Bin',[-.674, 0, .674]]
    FiveBin = ['5-Bin',[-.842, -0.253, 0.253, .842]]
    SixBin = ['6-Bin',[-0.967, -0.431, 0, 0.431, 0.967]]
    SevenBin = ['7-Bin',[-1.068, -0.566, -0.18, 0.18, 0.566, 1.068]]
    EightBin = ['8-Bin',[-1.15, -.674, -.319, 0, .319, .674, 1.15]]

    # Create a list for all bin arrays that will be used to build classification models.
    bins = []
    bins.append(TwoBin)
    bins.append(ThreeBin)
    bins.append(FourBin)
    bins.append(FiveBin)
    bins.append(SixBin)
    bins.append(SevenBin)
    bins.append(EightBin)

    return bins

    return bin

def BinData(df,BinList):

    TheBinsizeList = BinList[1]
    BinnedData =  np.asarray(DimensionalBinChoice(df, TheBinsizeList)) - 1
    return BinnedData

def Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList):
    # Standardizing the training set (apart from the test set).
    X_train = Standardize(X_train)

    # Bin the sum of scores of the training set.
    Y_train = np.asarray(DimensionalBinChoice(Y_train, TheBinsizeList))

    # Create an array of the training data.
    Y_train = Y_train - 1

    # Standardizing the testing set (apart from the training set).
    X_test = Standardize(X_test)

    # Bin the sum of scores of the testing set.
    Y_test = np.asarray(DimensionalBinChoice(Y_test, TheBinsizeList))

    # Create an array of the testing data.
    Y_test = Y_test - 1

    return X_train,X_test,Y_train,Y_test

def Kfold(model, X_Questions, y_SoS, TheBinsizeList):
    FoldValues = []
    kfold = KFold(n_splits=10, shuffle=False, random_state=None)
    counter = 0
    # Kfold
    for trainIndex, testIndex in kfold.split(X_Questions):
        X_train, X_test = X_Questions[trainIndex], X_Questions[testIndex]
        Y_train, Y_test = y_SoS[trainIndex], y_SoS[testIndex]

        X_train, X_test, Y_train, Y_test = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList)

        model.fit(X_train, Y_train, epochs=30, batch_size=30, verbose=0)
        predictions = model.predict(X_test)
        predictions_rounded = np.argmax(predictions, axis=1)
        Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions_rounded)
        PrintScores(Accuracy, Precision, Recall, Fscore)

        print("Kfold Iteration: " + str(counter))
        counter += 1

    return FoldValues

# This is a weighted averaging method for computing several different metrics.
def Scoring(TestTarget_Columns_Arr, predictions):
    Accuracy = accuracy_score(TestTarget_Columns_Arr, predictions)
    Precision = precision_score(TestTarget_Columns_Arr, predictions, average="weighted")
    Recall = recall_score(TestTarget_Columns_Arr, predictions, average="weighted")
    Fscore = f1_score(TestTarget_Columns_Arr, predictions, average="weighted")

    return Accuracy, Precision, Recall, Fscore

def PrintScores(Accuracy, Precision, Recall, Fscore, ModelName):
    print("For the " + str(ModelName) + " Model")
    print("Accuracy: " + str(Accuracy))
    print("Precision: " + str(Precision))
    print("Recall: " + str(Recall))
    print("F1: " + str(Fscore))