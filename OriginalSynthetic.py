import random
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import math
import tensorflow as tf
from pandas import DataFrame
import shap

def DimensionalityBinning_2(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[0] * StDev):
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  # 2.28% Person is Severly Depressed
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories


def SHAPInfo(model, X_MASQ_train_vals, X_MASQ_test_vals):
    explainer_Keras = shap.DeepExplainer(model, X_MASQ_train_vals) #DataFrames
    shap_values_Keras = explainer_Keras.shap_values(X_MASQ_test_vals) #DataFrames
    shap.summary_plot(shap_values_Keras, X_MASQ_test_vals) # Gives a bar graph of multiple classes

    shap.summary_plot(shap_values_Keras[0], X_MASQ_test_vals) # DataFrames
    return shap_values_Keras

def KerasModel(Dict):

    model = Sequential()
    model.add(Dense(Dict['L1Neurons'], input_dim=Dict['input_dim'], activation=Dict['activation']))
    # Uncomment below to add another hidden layer.
    # model.add(Dense(Dict['L2Neurons'], input_dim=Dict['L1Neurons'], activation=Dict['activation']))
    model.add(Dense(Dict['output'], activation=Dict['activation']))
    model.compile(loss=Dict['loss'], optimizer=Dict['optimizer'], metrics=[Dict['metrics']])

    return model

def Standardize(InDataframe):
    Dataframe = InDataframe.copy()
    col_names = Dataframe.columns
    features = Dataframe[col_names]
    Standardized = preprocessing.StandardScaler()
    scaler = Standardized.fit(features.values)
    features = scaler.transform(features.values)
    Dataframe[col_names] = features
    return Dataframe


def findweights():
            binarychoice = [0, 1]
            choice = random.choice(binarychoice)
            if choice == 0:
                tuple = (1, 2)
            elif choice == 1:
                tuple = (2, 1)
            else:
                print("Issues in findweights()")
            return tuple

def AnswerQuestions(numberofQuestions, DirichletProbs):
    test = np.random.choice([1, 2, 3, 4, 5], numberofQuestions, p=DirichletProbs)
    return test

def syntheticdata(choice):
    ListsofQuestions = []
    ListofVals = []
    if choice == 1:
        for x in range(Subjects):
            a, b = findweights()
            # Change a and b multipliers to change graph thickness
            [DirichletProbabilities] = np.random.dirichlet((a, 2 * a, 2 * (a ** 2 + b ** 2), 2 * b, b), size=1).round(
                10)
            AnsweredQuestion = AnswerQuestions(Questions, DirichletProbabilities)
            # ListofVals.append(DirichletProbabilities.tolist())
            ListsofQuestions.append(AnsweredQuestion.tolist())
    elif choice == 2:
        ListsofQuestions = [[random.randint(1, 5) for j in range(Questions)] for i in range(Subjects)]

    else:
        print("Issue in syntheticdata(choice) function")

    return ListsofQuestions

# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold
# data sample
# data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(n_splits=10, shuffle=False, random_state=None)
# enumerate splits


Questions = 30
Subjects = 2000
syntheticDatachoice = 1


SyntheticData = syntheticdata(syntheticDatachoice)
df = DataFrame.from_records(SyntheticData)
df["Sum_of_Scores"]=df.iloc[:,:].sum(axis=1)
Bin = [0]

ActivationFunction = 'sigmoid'
Layer1_Neurons = 2 #,20,30
LossFunction = 'categorical_crossentropy'
Optimizer= 'Adam'
FittingMetric = 'accuracy'

# Bins = len(Bin)+ 1
Bin_Length = len(Bin) + 1
ParameterDictionary = {'L1Neurons': Layer1_Neurons, 'input_dim': len(df.columns[:-1]),
                           'activation': ActivationFunction,
                           'L2Neurons': (math.ceil(Layer1_Neurons / 2)), 'output': Bin_Length,
                           'loss': LossFunction, 'optimizer': Optimizer,
                           'metrics': FittingMetric}



X = df[df.columns[:-1]].values
y = df[['Sum_of_Scores']].values
for trainIndex, testIndex in kfold.split(X):
    X_train, X_test = X[trainIndex], X[testIndex]
    y_train, y_test = y[trainIndex], y[testIndex]
    # If I only want to run this one time:
    # X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=.2)
    scaler = StandardScaler()
    # scaled = scaler.fit_transform(data)

    # X_train = scaler.fit_transform(X_train)
    # y_train = scaler.fit_transform(y_train)
    y_train_level = np.asarray(DimensionalityBinning_2(y_train, Bin))-1

    # X_test = scaler.fit_transform(X_test)
    # y_test = scaler.fit_transform(y_test)
    y_test_level = np.asarray(DimensionalityBinning_2(y_test, Bin))-1

    y_train_level_oneHot = tf.one_hot(y_train_level, Bin_Length)

    model = KerasModel(ParameterDictionary)

    fitModel = model.fit(X_train, y_train_level_oneHot, epochs=100, batch_size=10, verbose=0) # .decision_function(X_test)

    predictions = model.predict(X_test)
    predictions_rounded = np.argmax(predictions, axis=1)
    # predictions = (np.round_(predictions)).astype(int)

    # ModelScore = model.score(X_test, y_test_level)
    Accuracy_DF = accuracy_score(y_test_level, predictions_rounded)


    # print(ModelScore)
    print('Accuracy for DF: ' + str(Accuracy_DF))
    # for layer in model.layers: print(layer.get_config(), layer.get_weights())
