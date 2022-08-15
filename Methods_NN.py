#Good to go

# Import external packages.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import shap
from statistics import mean
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# Import local packages.
from DimensionalityBinning import DimensionalBinChoice
from SupportFunctions import PrintScores, Scoring, RoundandPercent, AverageList \
    , StdDevList, train_test_split, Bin_and_Standardize

# Create a neural network from a dictionary of inputs.
def KerasModel(Dict):
    # Create a sequential model.
    model = Sequential()
    # Add first hidden layer based on dictionary values.
    model.add(Dense(Dict['L1Neurons'], input_dim=Dict['input_dim'], activation=Dict['activation']))
    # Add second hidden layer based on dictionary values.
    model.add(Dense(Dict['L2Neurons'], input_dim=Dict['L1Neurons'], activation=Dict['activation']))
    # Add third hidden layer based on dictionary values.
    # model.add(Dense(Dict['L3Neurons'], input_dim=Dict['L2Neurons'], activation=Dict['activation']))
    # Add output layer.
    model.add(Dense(Dict['output'], activation=Dict['activation']))
    # Compile the model.
    model.compile(loss=Dict['loss'], optimizer=Dict['optimizer'], metrics=[Dict['metrics']])

    return model

def Model_NN(Questions, Bin_Length):
    from Methods_NN import NNParameters, KerasModel
    ParameterDictionary = NNParameters(Questions, Bin_Length)
    # y_train_level_oneHot = tf.one_hot(y_train_level, Bin_Length)
    model = KerasModel(ParameterDictionary)

    return model

# Create a neural network, train and test it using a subset of questions.
def Subset_Analysis_NN(FullQ_Train_Test_Split, ParameterDictionary_Subset, TheBinsizeList, Questionnaire_Subset):
    # FullQ_Train_Test_Split is the full set of questions which the
    X_Subset_train_std, X_Subset_test_std, y_Subset_train_std, y_Subset_test_std = FullQ_Train_Test_Split

    # Problem is, I'm passing in binned data from the full set of questions, I need to re-bin inside here so I can get individual
    # questions and subsets of questions.. so I just do binning in here as well based off of the individual question subset stds
    # and then go with y_train_SoS_std = X_Subset_train_std.sum(axis=1)
    X_Subset_train_std = X_Subset_train_std.loc[:, Questionnaire_Subset]
    # need to re-bin the data in here as I'm taking subsets!
    y_train_SoS_std = X_Subset_train_std.sum(axis=1)


    # Bin the sum of scores of the training set determined by the bin size in "TheBinSizeList".
    Subset_Binned_Level_y_train = DimensionalBinChoice(y_train_SoS_std, TheBinsizeList)

    # Create an array of the feature (question) values of questions defined in "Questionnaire_Subset".
    Train_Columns_Subset = np.asarray(X_Subset_train_std.loc[:, Questionnaire_Subset])
    # Create an array of target values of the training data comprised of the binned values from the subsets of questions
    # Sum of Scores (SoS).
    TrainTarget_Columns_Subset = np.asarray(Subset_Binned_Level_y_train)

    # start the binning from 0 for each of the subset binning.
    TrainTarget_Columns_Subset = TrainTarget_Columns_Subset - 1

    # Create the neural network model from the dictionary of (hyper)parameters.
    model_Subset = KerasModel(ParameterDictionary_Subset)
    # fit the model
    model_Subset.fit(Train_Columns_Subset, TrainTarget_Columns_Subset, epochs=30, batch_size=30, verbose=0)

    # Create an array of the testing feature data.
    Test_Columns_Subset = np.asarray(X_Subset_test_std.loc[:, Questionnaire_Subset])

    # Make a prediction from the test data.
    predictions_Subset = model_Subset.predict(Test_Columns_Subset)
    # Choose bin based on highest percentage probability.
    max_indices_Subset = np.argmax(predictions_Subset, axis=1)

    # Return the predictions.
    return max_indices_Subset

# Create and return a hyperparameter dictionary from the best found neural network (NN) hyperparameters via GridSearch.
def NNParameters(InputDim, OutputDim):
    LossFunction, Layer1_Neurons, Layer2_Neurons, ActivationFunction, Optimizer, FittingMetric = \
        'sparse_categorical_crossentropy', 50, 60, 'sigmoid', 'RMSProp', 'accuracy';

    # Create the dictionary of parameters.
    ParameterDictionary = {'input_dim': InputDim, 'activation': ActivationFunction,
                           'L1Neurons': Layer1_Neurons, 'L2Neurons': Layer2_Neurons,
                           'output': (OutputDim), 'loss': LossFunction, 'optimizer': Optimizer,
                           'metrics': FittingMetric}

    return ParameterDictionary

def Train_NN(X_train, Y_train, Questions, Bin_Length):
    model = Model_NN(Questions, Bin_Length)
    model.fit(X_train, Y_train, epochs=30, batch_size=30, verbose=1)
    return model

def Predict_NN(model, X_test, Y_test, TheBinsizeList):
    predictions = model.predict(X_test)
    predictions_rounded = np.argmax(predictions, axis=1)
    Accuracy, Precision, Recall, Fscore = Scoring(Y_test, predictions_rounded)
    ModelName = "NN"
    PrintScores(Accuracy, Precision, Recall, Fscore, ModelName)


def SHAPValues_NN(model, X_train, X_test):
    explainer_NN = shap.DeepExplainer(model, X_train)
    shap_values_NN = explainer_NN.shap_values(X_test)

    return explainer_NN, shap_values_NN

def SHAPPlots_NN(model, X_train, X_test, columns):
    col_val = columns[0]
    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = columns[:-1]
    X_train_df = pd.DataFrame(X_train)
    # test = X_test[5].columns
    X_train_df.columns = columns[:-1]
    explainer_NN, shap_values_NN = SHAPValues_NN(model, X_train, X_test)

    List = ["Summary_Single", "Summary_All", "Force_Datapoints_All", "Force_Bar", "Waterfall", "Decision", "Decision_Small"]
    # List = ["Decision_Small"]
    Bin_Configuration = 3
    Subject_Number = 712

    for Choice in List:
        if(Choice == "Summary_Single"):
            # As there are multiple shap value configurations, one for each bin, to get the scatter plot I have to choose a
            # particular configuration configuration for bin 0, so: shap_values[0] or bin 1 shap_values[1].
            # WORKS!
            shap.summary_plot(shap_values_NN[Bin_Configuration], X_test_df)

        elif(Choice=="Summary_All"):
            # If the shap value configuration is not selected e.g. shap_values instead of shap_values[0], then all the bin
            # classes are displayed as a part of a bar graph. Otherwise, for a single shap bin configuration, if a bar graph
            # is desired, then choose plot_type="bar".
            # WORKS!
            shap.summary_plot(shap_values_NN[Bin_Configuration], X_test_df, plot_type="bar", feature_names = X_test_df.columns)  # Gives a bar graph of multiple classes

        elif (Choice == "Force_Datapoints_All"):
            # Make the model that of a NN
            # # WORKS!
            plot = shap.force_plot(explainer_NN.expected_value[Bin_Configuration].numpy(), shap_values_NN[Bin_Configuration], X_test_df)
            shap.save_html("NN_All_Datapoints.htm", plot)

        elif (Choice == "Force_Bar"):
            # WORKS!
            plot = shap.force_plot(explainer_NN.expected_value[Bin_Configuration].numpy(), shap_values_NN[Bin_Configuration][10, :], X_test_df.iloc[10, :])
            shap.save_html("NN_Individual_Datapoint.htm", plot)

        elif (Choice == "Waterfall"):
            # Note, the difference in values between this and SVM KernelExplainer might be because I added a \
            # [Subject_Number] for an individual subject, but this also changes between runs because the order is randomized.
            shap.plots._waterfall.waterfall_legacy(explainer_NN.expected_value[Bin_Configuration].numpy(), shap_values_NN[Bin_Configuration][Subject_Number],
                                                   feature_names=X_test_df.columns)

        elif (Choice == "Decision"):
            shap.decision_plot(explainer_NN.expected_value[Bin_Configuration].numpy(), shap_values_NN[Bin_Configuration][Subject_Number], features=X_test_df.iloc[0, :],
                               feature_names=X_test_df.columns.tolist())

        elif (Choice == "Decision_Small"):
            # plot = shap.dependence_plot("PSWQ_4", shap_values_NN[Bin_Configuration], X_test_df[5], interaction_index=X_test_df[7], \
            #                             cmap=plt.get_cmap("cool"))
            # shap.save_html("Decision_Small.htm", plot)
            shap.dependence_plot(col_val, shap_values_NN[0], X_test_df)

        else:
            print("Selection Not Found.")

# This method finds the prediction for each individual question.
def Compute_Individual_Predictions(Full_dataset, bins, NumTrials = 2):
    from Create_or_Choose_Dataset import Create_Synthetic_Data

    # Create and map the name 'BinSize' to the first column.
    # mapping = {Full_dataset.columns[0]: 'BinSize'}
    #
    # # Rename the columns to be of that mapped.
    # Full_dataset= Full_dataset.rename(columns=mapping)

    X = Full_dataset.iloc[:, 0:-1]
    Y = Full_dataset.iloc[:, -1:]

    X_arr = X.values
    Y_arr = Y.values


    df_Col_Names = Full_dataset.iloc[:, :-1].columns

    # Define hyperparameters for the neural network type which will be used to create models for all classifications.


    # Create lists for the metrics we want to keep track of.
    FullQuestionnaireMetrics = ['Model', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1']
    BriefQuestionnaireMetrics = ['SubsetInfo', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1']
    FullQuestionnaire_StDevMetrics = ['Accuracy_StDev', 'Precision_StDev', 'Recall_StDev', 'F1_StDev']

    # Create dataframes for holding output data for both the whole questionnaire as well as each of the brief questionnaires
    # (Subsets).
    ScoringMetricsConcat_DF_Subset_Ave = pd.DataFrame(columns=BriefQuestionnaireMetrics)
    ScoringMetricsStDevConcat_DF = pd.DataFrame(columns=FullQuestionnaire_StDevMetrics)

    Individual_Sample_Metrics_DF = pd.DataFrame(columns=X.columns)
    Individual_Sample_Metrics_DF["Trial"] = 0
    Individual_Sample_Metrics_DF["Bin"] = 0

    Total_SingleQ_Accuracy_DF = pd.DataFrame(columns=Individual_Sample_Metrics_DF.columns)

    TestDF = pd.DataFrame(columns=BriefQuestionnaireMetrics)
    BinNames = []

    # Note - the same subject train/test sample ordering is used for each of the bins, though it is shuffled for each of
    # the trials in which to build a new model.
    for BinsizeName, TheBinsizeList in bins:
        print("For the BinSize " + BinsizeName)
        BinNames.append(BinsizeName)

        # These lists will hold the four tracked metrics for classifications over the bin array using all questions.
        Accuracy_Subset_AllQuestions_List, Precision_Subset_AllQuestions_List, Recall_Subset_AllQuestions_List, \
        Fscore_Subset_AllQuestions_List = [], [], [], [];

        # These lists hold the set of scoring metrics for the full and abbreviated questionnaires.
        ScoringMetrics_Ave, ScoringMetrics_Subset_Ave, StDevMetrics_Ave = [], [], [];

        ScoringMetrics_Subset = []

        # Holds the classification metrics for the full and abbreviated metrics at the model level (before averaging).
        AccuracyList, PrecisionList, RecallList, FscoreList = [], [], [], [];
        AccuracyArr_Subset, PrecisionArr_Subset, RecallArr_Subset, FscoreArr_Subset = [], [], [], [];

        # Holds the pre-averaged metrics for the iterative question addition.
        AccuracySubset_List, PrecisionSubset_List, RecallSubset_List, FscoreSubset_List = [], [], [], [];

        # Do the computations for the number trials (which the metrics will be averaged over).
        for TrialNumber in range(NumTrials):
            print('Trial Number ' + str(
                TrialNumber + 1) + ' For the Subject Subset counter ' + " In the Binsize " + BinsizeName)

            # Training and testing split. - This will create a different randomized ordering, though this might not be
            # the best as we could get a different SHAP ordering from what might be found by a different train-set if
            # it were used in SHAP_averaging. However, we do get models built on different train sets so we avoid
            # the potential of finding a particular train/test set that models well.
            X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, shuffle=True, test_size=.2)
            X_train_std, X_test_std, Y_train_binned, Y_test_binned = Bin_and_Standardize(X_train,X_test,Y_train,Y_test, TheBinsizeList)

            # define the input and output layer sizes and create the dictionary of neural network parameters.
            inDim = len(X.columns)
            outDim = len(TheBinsizeList) + 1
            ParameterDictionary = NNParameters(inDim, outDim)

            # Create the neural network.
            model = KerasModel(ParameterDictionary)
            # Fit the neural network to the training data. We already standardized and binned it above.
            model.fit(X_train_std, Y_train_binned, epochs=30, batch_size=32, verbose=0)

            # Make a prediction from the test data.
            predictions = model.predict(X_test_std)

            # Choose bin based on highest percentage probability.
            max_indices = np.argmax(predictions, axis=1)

            # Compute metrics by comparing actual (Y_test) vs. predicted (max_indices).
            Accuracy, Precision, Recall, Fscore = Scoring(Y_test_binned, max_indices)

            # These lists will hold the four tracked metrics for classifications over the bin array using subsets of questions.
            Accuracy_Subset_AllQuestions, Precision_Subset_AllQuestions, Recall_Subset_AllQuestions, \
            Fscore_Subset_AllQuestions = [], [], [], [];

            Accuracy_per_Trial_by_Bin = []

            # Holds the list of questions to be tested.
            Subset_to_Test = []

            # Here we convert to dataframes so it is easy to order subsets based on SHAP Averaged ordering
            X_train_std_df = pd.DataFrame(X_train_std, columns=X.columns)
            X_test_std_df = pd.DataFrame(X_test_std, columns=X.columns)
            Y_train_binned_df = pd.DataFrame(Y_train_binned, columns=['Bin'])
            Y_test_binned_df = pd.DataFrame(Y_test_binned, columns=['Bin'])

            # Create a list from the training and testing splits.
            Train_Test_List = [X_train_std_df, X_test_std_df, Y_train_binned_df, Y_test_binned_df]

            # The first parameter is the number of questions - 1 as I'm testing each question individually
            ParameterDictionary_Subset = NNParameters(1, len(TheBinsizeList) + 1)
            # Build abbreviated questionnaires, question by question:
            for Question in X.columns:
                # Make the subset just equal to the new question
                Subset_to_Test = [Question]

                # Choose bin based on highest percentage probability.
                max_indices_Subset = Subset_Analysis_NN(Train_Test_List, ParameterDictionary_Subset, TheBinsizeList,
                                                        Subset_to_Test)

                # Note that we are testing the Subset predictions (max_indices_Subset) against the full measure
                # predictions (Y_test). We don't want to test the Target Columns of the subset against the
                # Subset predictions as we want to compare the subset predictions against the full measure target values.

                # Compute the metrics of each of the subsets (abbreviated questionnaires), iteratively (question by question)
                # and append those values to an array holding the values for each of the metrics. For example, for a 3
                # question abbreviated questionnaire, there will be one, two, and finally three values in each of the
                # lists being appended.

                # Score the subsets (abbreviated questionnaires) against total number of questions.
                Accuracy_Subset, Precision_Subset, Recall_Subset, Fscore_Subset = Scoring(Y_test_binned, max_indices_Subset)
                Individual_Sample_Metrics_DF[Question] = Accuracy_Subset

                # Create lists to hold abbreviated questionnaire (Subset) metrics for each of the trials.
                Accuracy_Subset_AllQuestions.append(Accuracy_Subset)
                Precision_Subset_AllQuestions.append(Precision_Subset)
                Recall_Subset_AllQuestions.append(Recall_Subset)
                Fscore_Subset_AllQuestions.append(Fscore_Subset)

            # Append the metrics for the full questionnaires for each trial to an list.
            AccuracyList.append(Accuracy)
            PrecisionList.append(Precision)
            RecallList.append(Recall)
            FscoreList.append(Fscore)


            # Append the lists for each metric (for each brief) for each trial to a list.
            Accuracy_Subset_AllQuestions_List.append(Accuracy_Subset_AllQuestions)
            Precision_Subset_AllQuestions_List.append(Precision_Subset_AllQuestions)
            Recall_Subset_AllQuestions_List.append(Recall_Subset_AllQuestions)
            Fscore_Subset_AllQuestions_List.append(Fscore_Subset_AllQuestions)

            Accuracy_per_Trial_by_Bin = Accuracy_Subset_AllQuestions[:]
            Accuracy_per_Trial_by_Bin.append(TrialNumber + 1)
            Accuracy_per_Trial_by_Bin.append(len(TheBinsizeList) + 1)
            TrialAccuracy_DF = pd.DataFrame([Accuracy_per_Trial_by_Bin])
            Individual_Sample_Metrics_DF[Individual_Sample_Metrics_DF.columns] = TrialAccuracy_DF

            Total_SingleQ_Accuracy_DF = \
            pd.concat([Total_SingleQ_Accuracy_DF, Individual_Sample_Metrics_DF], axis=0)

            # release the memory from building the model.
            K.clear_session()

        # Average the metrics for each of the full questionnaires over all of the trials.
        AccuracyAve = mean(AccuracyList)
        PrecisionAve = mean(PrecisionList)
        RecallAve = mean(RecallList)
        FscoreAve = mean(FscoreList)

        # Average the lists of metrics for each of the brief questionnaires over all of the trials.
        AccuracyAve_Subset = RoundandPercent(AverageList(Accuracy_Subset_AllQuestions_List))
        StdevArr_Subset_Accuracy = RoundandPercent(StdDevList(Accuracy_Subset_AllQuestions_List))
        PrecisionAve_Subset = RoundandPercent(AverageList(Precision_Subset_AllQuestions_List))
        StdevArr_Subset_Precision = RoundandPercent(StdDevList(Precision_Subset_AllQuestions_List))
        RecallAve_Subset = RoundandPercent(AverageList(Recall_Subset_AllQuestions_List))
        StdevArr_Subset_Recall = RoundandPercent(StdDevList(Recall_Subset_AllQuestions_List))
        FscoreAve_Subset = RoundandPercent(AverageList(Fscore_Subset_AllQuestions_List))
        StdevArr_Subset_Fscore = RoundandPercent(StdDevList(Fscore_Subset_AllQuestions_List))

        # Retrieve hyperparameter values.
        InputDimension = ParameterDictionary['input_dim']
        OutputDimension = ParameterDictionary['output']
        ActivationFunction = ParameterDictionary['activation']
        FirstLayerNeurons = ParameterDictionary['L1Neurons']
        SecondLayerNeurons = ParameterDictionary['L2Neurons']
        Loss_Function = ParameterDictionary['loss']
        TheOptimizer = ParameterDictionary['optimizer']
        Fitting_Metric = ParameterDictionary['metrics']

        # Create a string of hyperparameter values and their descriptions.
        ModelInfo = 'Activation: ' + str(ActivationFunction) + ', Layer 1: ' + str(
            FirstLayerNeurons) + ', LossFunction: ' + \
                    Loss_Function + ', Optimizer: ' + TheOptimizer + ', FittingMetric: ' + Fitting_Metric + ', Questions: ' + \
                    str(InputDimension) + ', OutputBins: ' + str(OutputDimension)

        # For each average of trials, Append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics_Ave.append(
            [ModelInfo, len(TheBinsizeList) + 1, AccuracyAve, PrecisionAve, RecallAve,
             FscoreAve])

        # Create a string for subset info.
        Subset_Info = 'Questions: ' + \
                      str(len(Subset_to_Test)) + ', OutputBins: ' + str(len(TheBinsizeList) + 1)

        # Here I can create a ScoringMetrics_Subset and append each of the metric lists containing all trials.
        ScoringMetrics_Subset.append([Subset_Info,len(TheBinsizeList)+ 1, Accuracy_Subset_AllQuestions_List, \
                Precision_Subset_AllQuestions_List,Recall_Subset_AllQuestions_List, Fscore_Subset_AllQuestions_List])

        # For each average of trials, append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics_Subset_Ave.append(
            [Subset_Info, len(TheBinsizeList) + 1, tuple(AccuracyAve_Subset),
             tuple(PrecisionAve_Subset), tuple(RecallAve_Subset), tuple(FscoreAve_Subset)])

        # Append all of the standard deviations for the various metrics to a list.
        StDevMetrics_Ave.append([tuple(StdevArr_Subset_Accuracy), tuple(StdevArr_Subset_Precision),
                             tuple(StdevArr_Subset_Recall), tuple(StdevArr_Subset_Fscore)])

        # Create dataframes for both the full and abbreviated questionnaire metrics and the metric standard deviations.
        SubjectsandQuestions_DF_Subset = pd.DataFrame.from_records(ScoringMetrics_Subset_Ave,
                                                                   columns=BriefQuestionnaireMetrics)
        SubjectsandQuestions_DF = pd.DataFrame.from_records(ScoringMetrics_Subset, columns=BriefQuestionnaireMetrics)
        print(SubjectsandQuestions_DF)
        TestDF = pd.concat([SubjectsandQuestions_DF_Subset, TestDF]).copy()
        ScoringMetricsConcat_DF_Subset_Ave = pd.concat([ScoringMetricsConcat_DF_Subset_Ave, SubjectsandQuestions_DF_Subset],
                                                   axis=0)

    # Append the full and abbreviated questionnaires for each new bin array.
    # ScoringMetricsConcat_DF = pd.concat([ScoringMetricsConcat_DF, SubjectsandQuestions_DF], axis=0)
    # TestDF = pd.concat([SubjectsandQuestions_DF_Subset, TestDF])
    # print(TestDF)
    # print(ScoringMetricsConcat_DF_Subset_Ave)
    # ScoringMetricsStDevConcat_DF = pd.concat([ScoringMetricsStDevConcat_DF, StDevMetrics_DF], axis=0)
    # ScoringMetricsConcat_DF_Subset_Ave = pd.concat([ScoringMetricsConcat_DF_Subset_Ave, SubjectsandQuestions_DF_Subset], axis=0)

    print(ScoringMetricsConcat_DF_Subset_Ave)
    # print(ScoringMetricsStDevConcat_DF)

    QuestionIterator = list(range(1, len(X.columns) + 1))

    # Add all metrics to the graph.
    # Yaxis = ["Accuracy", "Precision", "Recall", "F1"]
    # # Graph the results.
    # for l in Yaxis:
    #     df_long = ScoringMetricsConcat_DF_Subset_Ave.explode(l).reset_index()
    #     df_long.drop('index', axis=1, inplace=True)
    #     df_long['Questions'] = np.tile(QuestionIterator, len(ScoringMetricsConcat_DF_Subset_Ave))
    #     df_long[l] = df_long[l].astype(float)
    #     g = sns.relplot(x='Questions', y=l, hue="Binsize",
    #                     data=df_long, height=5, aspect=.8, kind='line')
    #     g._legend.remove()
    #     g.fig.suptitle(l + " Score")
    #     g.fig.subplots_adjust(top=.95)
    #     g.ax.set_xlabel('Questions', fontsize=12)
    #     g.ax.set_ylabel(l, fontsize=12)
    #     plt.xticks(QuestionIterator)
    #     legend_title = 'Bins/Levels'
    #     g._legend.set_title(legend_title)
    #     # Create new labels for the legend.
    #     Binsize_list = ScoringMetricsConcat_DF_Subset_Ave['Binsize'].tolist()
    #     new_labels = [str(x) for x in BinsizeName]
    #     for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    #     plt.legend(title='Configurations', loc='lower right', labels=new_labels)
    #     g.tight_layout()

    plt.show()

    return ScoringMetricsConcat_DF_Subset_Ave, Total_SingleQ_Accuracy_DF

def Compute_Subsets_NN(Full_dataset, bins, NumTrials = 5):
    from Create_or_Choose_Dataset import Create_Synthetic_Data

    # Create and map the name 'BinSize' to the first column.
    # mapping = {Full_dataset.columns[0]: 'BinSize'}
    #
    # # Rename the columns to be of that mapped.
    # Full_dataset= Full_dataset.rename(columns=mapping)

    X = Full_dataset.iloc[:, 0:-1]
    Y = Full_dataset.iloc[:, -1:]

    X_arr = X.values
    Y_arr = Y.values

    df_Col_Names = Full_dataset.iloc[:,:-1].columns


    # # Length of first element will show how many questions/features/columns there are.
    # Num_Questions = len(X_train[0])
    # # Sum of array elements in X_train and X_test give the total number of subjects
    # NumSubjects = len(X_train) + len(X_test)
    BinNames = []

    DummyInputDim, DummyOutputDim = 0, 0;

    # Define hyperparameters for the neural network type which will be used to create models for all classifications.
    ParameterDictionary_Subset = NNParameters(DummyInputDim, DummyOutputDim)

    # Create lists for the metrics we want to keep track of.
    FullQuestionnaireMetrics = ['Model', 'Set', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1']
    BriefQuestionnaireMetrics = ['SubsetInfo', 'Set', 'Binsize', 'Accuracy', 'Precision', 'Recall', 'F1']
    FullQuestionnaire_StDevMetrics = ['Accuracy_StDev', 'Precision_StDev', 'Recall_StDev', 'F1_StDev']

    # Create dataframes for holding output data for both the whole questionnaire as well as each of the brief questionnaires
    # (Subsets).
    ScoringMetricsConcat_DF = pd.DataFrame(columns=FullQuestionnaireMetrics)
    ScoringMetricsConcat_DF_Subset = pd.DataFrame(columns=BriefQuestionnaireMetrics)
    ScoringMetricsStDevConcat_DF = pd.DataFrame(columns=FullQuestionnaire_StDevMetrics)
    SubjectsandQuestions_DF, SubjectsandQuestions_DF_Subset = pd.DataFrame(), pd.DataFrame()
    TestDF = pd.DataFrame(columns=BriefQuestionnaireMetrics)


    Bin_Iterator = 0

    # Note - the same subject train/test sample ordering is used for each of the bins, though it is shuffled for each of
    # the trials in which to build a new model.
    for BinsizeName, TheBinsizeList in bins:
        print("For the BinSize " + BinsizeName)
        BinNames.append(BinsizeName)

        # model = Train_NN(X_train, Y_train, Num_Questions, len(TheBinsizeList) + 1)

        # For each Bin STD configuration in the bins list we are taking the particular train/test sets to create a SHAP
        # ordering.

        # SHAP_Averaging works to return the question ordering with each of their averaged SHAP values for each of the questions.
        Initial_SHAP_Orderings_df = SHAP_Averaging(X_arr, Y_arr, bins[Bin_Iterator], df_Col_Names, 10)

        # Create the NN here and try to predict Synthetic Data

        # Extract the row of questions (from file) associated to each bin array.
        RowofQuestions = Initial_SHAP_Orderings_df.columns

        # Create a list and pull off the outer [ ] brackets.
        Subset_TrainAndTest = RowofQuestions.tolist()

        # Remove None values in list.
        Subset_TrainAndTest = list(filter(None, Subset_TrainAndTest))

        # Update the output size of the neural network to be that of the size of the current bin array.
        Output = {'output': len(TheBinsizeList) + 1}
        ParameterDictionary_Subset.update(Output)

        # These lists will hold the four tracked metrics for classifications over the bin array using all questions.
        Accuracy_Subset_AllQuestions_List, Precision_Subset_AllQuestions_List, Recall_Subset_AllQuestions_List, \
        Fscore_Subset_AllQuestions_List = [], [], [], [];

        # These lists hold the set of scoring metrics for the full and abbreviated questionnaires.
        ScoringMetrics, ScoringMetrics_Subset, StDevMetrics = [], [], [];

        # Holds the classification metrics for the full and abbreviated metrics at the model level (before averaging).
        AccuracyList, PrecisionList, RecallList, FscoreList = [], [], [], [];
        AccuracyArr_Subset, PrecisionArr_Subset, RecallArr_Subset, FscoreArr_Subset = [], [], [], [];
        Bin_Iterator = Bin_Iterator + 1


        # Do the computations for the number trials (which the metrics will be averaged over).
        for TrialNumber in range(NumTrials):
            print('Trial Number ' + str(TrialNumber + 1) + ' For the Subject Subset counter ' + " In the Binsize " + BinsizeName)
            X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=.2, shuffle=True)
            X_train_std, X_test_std, Y_train_binned, Y_test_binned = Bin_and_Standardize(X_train, X_test, Y_train, Y_test, TheBinsizeList)

            # Training and testing split. - This will create a different randomized ordering, though this might not be
            # the best as we could get a different SHAP ordering from what might be found by a different train-set if
            # it were used in SHAP_averaging. However, we do get models built on different train sets so we avoid
            # the potential of finding a particular train/test set that models well.
            # X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, shuffle=True, test_size=.2)
            # X_train_std, X_test_std, Y_train_binned, Y_test_binned = Bin_and_Standardize(X_train,X_test,Y_train,Y_test, TheBinsizeList)

            # define the input and output layer sizes and create the dictionary of neural network parameters.
            inDim = len(X.columns)
            outDim = len(TheBinsizeList) + 1
            ParameterDictionary = NNParameters(inDim, outDim)

            # Create the neural network.
            model = KerasModel(ParameterDictionary)
            # Fit the neural network to the training data. We already standardized and binned it above.
            model.fit(X_train_std, Y_train_binned, epochs=30, batch_size=32, verbose=0)

            # Make a prediction from the test data.
            predictions = model.predict(X_test_std)

            # Choose bin based on highest percentage probability.
            max_indices = np.argmax(predictions, axis=1)

            # Compute metrics by comparing actual (Y_test) vs. predicted (max_indices).
            Accuracy, Precision, Recall, Fscore = Scoring(Y_test_binned, max_indices)

            # These lists will hold the four tracked metrics for classifications over the bin array using subsets of questions.
            Accuracy_Subset_AllQuestions, Precision_Subset_AllQuestions, Recall_Subset_AllQuestions, \
            Fscore_Subset_AllQuestions = [], [], [], [];


            # Holds the list of questions to be tested.
            Subset_to_Test = []

            # Here we convert to dataframes so it is easy to order subsets based on SHAP Averaged ordering
            X_train_std_df = pd.DataFrame(X_train_std, columns=X.columns)
            X_test_std_df = pd.DataFrame(X_test_std, columns=X.columns)
            Y_train_binned_df = pd.DataFrame(Y_train_binned, columns=['Bin'])
            Y_test_binned_df = pd.DataFrame(Y_test_binned, columns=['Bin'])

            # Create a list from the training and testing splits.
            Train_Test_List = [X_train_std_df, X_test_std_df, Y_train_binned_df, Y_test_binned_df]


            # Build abbreviated questionnaires, question by question:
            for Question in Subset_TrainAndTest:
                # Iteratively build a larger list Question by question.
                Subset_to_Test.append(Question)

                # Update the input dimension keyword in the dictionary based on how large the subset of questions is.
                Input = {'input_dim': len(Subset_to_Test)}
                ParameterDictionary_Subset.update(Input)

                # Choose bin based on highest percentage probability.
                max_indices_Subset = Subset_Analysis_NN(Train_Test_List, ParameterDictionary_Subset, TheBinsizeList,
                                                        Subset_to_Test)

                # Note that we are testing the Subset predictions (max_indices_Subset) against the full measure
                # predictions (Y_test). We don't want to test the Target Columns of the subset against the
                # Subset predictions as we want to compare the subset predictions against the full measure target values.

                # Compute the metrics of each of the subsets (abbreviated questionnaires), iteratively (question by question)
                # and append those values to an array holding the values for each of the metrics. For example, for a 3
                # question abbreviated questionnaire, there will be one, two, and finally three values in each of the
                # lists being appended.

                # Score the subsets (abbreviated questionnaires) against total number of questions.
                Accuracy_Subset, Precision_Subset, Recall_Subset, Fscore_Subset = \
                    Scoring(Y_test_binned, max_indices_Subset)

                # Create lists to hold abbreviated questionnaire (Subset) metrics for each of the trials.
                Accuracy_Subset_AllQuestions.append(Accuracy_Subset)
                Precision_Subset_AllQuestions.append(Precision_Subset)
                Recall_Subset_AllQuestions.append(Recall_Subset)
                Fscore_Subset_AllQuestions.append(Fscore_Subset)

            # Append the metrics for the full questionnaires for each trial to an list.
            AccuracyList.append(Accuracy)
            PrecisionList.append(Precision)
            RecallList.append(Recall)
            FscoreList.append(Fscore)

            # Append the lists for each metric (for each brief) for each trial to a list.
            Accuracy_Subset_AllQuestions_List.append(Accuracy_Subset_AllQuestions)
            Precision_Subset_AllQuestions_List.append(Precision_Subset_AllQuestions)
            Recall_Subset_AllQuestions_List.append(Recall_Subset_AllQuestions)
            Fscore_Subset_AllQuestions_List.append(Fscore_Subset_AllQuestions)

            # release the memory from building the model.
            K.clear_session()


        # Average the metrics for each of the full questionnaires over all of the trials.
        AccuracyAve = mean(AccuracyList)
        PrecisionAve = mean(PrecisionList)
        RecallAve = mean(RecallList)
        FscoreAve = mean(FscoreList)

        # Average the lists of metrics for each of the brief questionnaires over all of the trials.
        AccuracyAve_Subset = RoundandPercent(AverageList(Accuracy_Subset_AllQuestions_List))
        StdevArr_Subset_Accuracy = RoundandPercent(StdDevList(Accuracy_Subset_AllQuestions_List))
        PrecisionAve_Subset = RoundandPercent(AverageList(Precision_Subset_AllQuestions_List))
        StdevArr_Subset_Precision = RoundandPercent(StdDevList(Precision_Subset_AllQuestions_List))
        RecallAve_Subset = RoundandPercent(AverageList(Recall_Subset_AllQuestions_List))
        StdevArr_Subset_Recall = RoundandPercent(StdDevList(Recall_Subset_AllQuestions_List))
        FscoreAve_Subset = RoundandPercent(AverageList(Fscore_Subset_AllQuestions_List))
        StdevArr_Subset_Fscore = RoundandPercent(StdDevList(Fscore_Subset_AllQuestions_List))

        # Retrieve hyperparameter values.
        InputDimension = ParameterDictionary['input_dim']
        OutputDimension = ParameterDictionary['output']
        ActivationFunction = ParameterDictionary['activation']
        FirstLayerNeurons = ParameterDictionary['L1Neurons']
        SecondLayerNeurons = ParameterDictionary['L2Neurons']
        Loss_Function = ParameterDictionary['loss']
        TheOptimizer = ParameterDictionary['optimizer']
        Fitting_Metric = ParameterDictionary['metrics']

        # Create a string of hyperparameter values and their descriptions.
        ModelInfo = 'Activation: ' + str(ActivationFunction) + ', Layer 1: ' + str(
            FirstLayerNeurons) + ', LossFunction: ' + \
                    Loss_Function + ', Optimizer: ' + TheOptimizer + ', FittingMetric: ' + Fitting_Metric + ', Questions: ' + \
                    str(InputDimension) + ', OutputBins: ' + str(OutputDimension)

        # For each average of trials, Append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics.append(
            [ModelInfo, Bin_Iterator, len(TheBinsizeList) + 1, AccuracyAve, PrecisionAve, RecallAve,
             FscoreAve])

        # Create a string for subset info.
        Subset_Info = 'Questions: ' + \
                      str(len(Subset_TrainAndTest)) + ', OutputBins: ' + str(len(TheBinsizeList) + 1)

        # For each average of trials, append the scoring metrics and model info for the full questionnaire.
        ScoringMetrics_Subset.append(
            [Subset_Info, Bin_Iterator, len(TheBinsizeList) + 1, tuple(AccuracyAve_Subset),
             tuple(PrecisionAve_Subset), tuple(RecallAve_Subset), tuple(FscoreAve_Subset)])

        # Append all of the standard deviations for the various metrics to a list.
        StDevMetrics.append([tuple(StdevArr_Subset_Accuracy), tuple(StdevArr_Subset_Precision),
                             tuple(StdevArr_Subset_Recall), tuple(StdevArr_Subset_Fscore)])

        # Create dataframes for both the full and abbreviated questionnaire metrics and the metric standard deviations.
        SubjectsandQuestions_DF = pd.DataFrame.from_records(ScoringMetrics, columns=FullQuestionnaireMetrics)
        SubjectsandQuestions_DF_Subset = pd.DataFrame.from_records(ScoringMetrics_Subset,
                                                                   columns=BriefQuestionnaireMetrics)
        StDevMetrics_DF = pd.DataFrame.from_records(StDevMetrics, columns=FullQuestionnaire_StDevMetrics)
        # ScoringMetricsConcat_DF_Subset.append(SubjectsandQuestions_DF_Subset, ignore_index=True)
        TestDF = pd.concat([SubjectsandQuestions_DF_Subset, TestDF]).copy()
        ScoringMetricsConcat_DF_Subset = pd.concat([ScoringMetricsConcat_DF_Subset, SubjectsandQuestions_DF_Subset],
                                                   axis=0)


    # Append the full and abbreviated questionnaires for each new bin array.
    # ScoringMetricsConcat_DF = pd.concat([ScoringMetricsConcat_DF, SubjectsandQuestions_DF], axis=0)
    # TestDF = pd.concat([SubjectsandQuestions_DF_Subset, TestDF])
    # print(TestDF)
    # print(ScoringMetricsConcat_DF_Subset)
    ScoringMetricsStDevConcat_DF = pd.concat([ScoringMetricsStDevConcat_DF, StDevMetrics_DF], axis=0)
    # ScoringMetricsConcat_DF_Subset = pd.concat([ScoringMetricsConcat_DF_Subset, SubjectsandQuestions_DF_Subset], axis=0)

    print(ScoringMetricsConcat_DF_Subset)
    print(ScoringMetricsStDevConcat_DF)

    # Create a list of numbers to denote questions in the graph.
    QuestionIterator = list(range(1, len(Subset_TrainAndTest) + 1))

    # Add all metrics to the graph.
    Yaxis = ["Accuracy", "Precision", "Recall", "F1"]
    # Graph the results.
    for l in Yaxis:
        df_long = ScoringMetricsConcat_DF_Subset.explode(l).reset_index()
        df_long.drop('index', axis=1, inplace=True)
        df_long['Questions'] = np.tile(QuestionIterator, len(ScoringMetricsConcat_DF_Subset))
        df_long[l] = df_long[l].astype(float)
        g = sns.relplot(x='Questions', y=l, hue="Set",
                        data=df_long, height=5, aspect=.8, kind='line')
        g._legend.remove()
        g.fig.suptitle(l + " Score")
        g.fig.subplots_adjust(top=.95)
        g.ax.set_xlabel('Questions', fontsize=12)
        g.ax.set_ylabel(l, fontsize=12)
        plt.xticks(QuestionIterator)
        legend_title = 'Bins/Levels'
        g._legend.set_title(legend_title)
        # Create new labels for the legend.
        Binsize_list = ScoringMetricsConcat_DF_Subset['Binsize'].tolist()
        new_labels = [str(x) for x in BinNames]
        for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
        plt.legend(title='Configurations', loc='lower right', labels=new_labels)
        g.tight_layout()

    plt.show()


def SHAP_Averaging(X_arr,Y_arr, TheBinsizelist, Col_Names, NumTrials=2):
    X_train, X_test, Y_train, Y_test = train_test_split(X_arr, Y_arr, test_size=.2)
    Binsize_length = TheBinsizelist[1]

    # Create a dataframe from the independent features.
    df = pd.DataFrame(columns=Col_Names)
    # Create a column for the trial number.
    df.insert(0, 'Trial', 0)

    InputDim = len(df.iloc[:, 0:-1].columns)
    OutputDim = len(Binsize_length) + 1

    # Creates a dictionary of hyperparameters and input and output sizes.
    ParameterDictionary = NNParameters(InputDim, OutputDim)
    model_Keras = KerasModel(ParameterDictionary)

    # Create a list for bin-configuration level names
    ConfigurationList = []
    for i in range(OutputDim):
        ConfigurationList.append("ConfigurationLevel_" + str(i + 1))


    # Number of Trials to perform.
    # NumTrials = 2
    # Do the computations for the number trials (which the metrics will be averaged over).
    for TrialNumber in range(NumTrials):
        print("SHAP Averaging Trial Number: " + str(TrialNumber))
        # Training and testing split.
        Train_Columns_Arr, Test_Columns_Arr, TrainTarget_Columns_Arr, TestTarget_Columns_Arr = \
            Bin_and_Standardize(X_train, X_test, Y_train, Y_test, Binsize_length)

        # Fit the neural network to the training data.
        model_Keras.fit(Train_Columns_Arr, TrainTarget_Columns_Arr, epochs=30, batch_size=32, verbose=0)
        # Make a prediction from the test data.
        predictions = model_Keras.predict(Test_Columns_Arr)
        # Choose bin based on highest percentage probability.
        predictions = np.argmax(predictions, axis=1)

        # Compute metrics by comparing actual (TestTarget) vs. predicted (predictions).
        Accuracy, Precision, Recall, Fscore = Scoring(TestTarget_Columns_Arr, predictions)

        # Calculate SHAP values.
        explainer_NN, shap_values_NN = SHAPValues_NN(model_Keras, Train_Columns_Arr, Test_Columns_Arr)

        shapoutputsize = len(shap_values_NN)
        for i in range(shapoutputsize):
            # First, averaging is being done over each of the SHAP values for each question in the *shap_values_NN[i].
            # For example, if there are 500 subjects in the training set the first elem will be all of their SHAP values
            # for question 0, the second elem will be the 500 SHAP values for question 1 etc. This is being done for each
            # of the outputs in shapouputsize which means that for 4 bins there will be four SHAP levels for shap_values_NN[i]
            # where i = 0 to 3. Now in general there are the k bins/levels for each fo the n trials. These k * n are all
            # inserted in the dataframe in an order such that the levels associated for a particular trial are subsequently
            # displayed.
            # list_of_ave_val = []
            # for elem in zip(*shap_values_NN[i]):
            #     # Sum over all SHAP values for each subject for the particular bin-configuration. NOTE! There is an absolute value!
            #     sum_val = sum(np.abs(elem))
            #     # Number of subjects
            #     len_val = len(elem)
            #     # Average SHAP value per subject per particular bin-configuratoin
            #     ave_val = sum_val / len_val
            #     list_of_ave_val(ave_val)

            # print([sum(np.abs(elem)) / len(elem) for elem in zip(*shap_values_NN[i])])
            # print(list_of_ave_val)
            df.loc[(len(Binsize_length) + 1) * TrialNumber + i] = ['Trial_' + str(TrialNumber + 1) + " " + str(i)] \
                                                                + [sum(np.abs(elem)) / len(elem) for elem in
                                                                     zip(*shap_values_NN[i])]

    # Combined_SHAP_df holds each of averaged SHAP values.
    # Combined_SHAP_df = pd.DataFrame(index=ConfigurationList)
    Combined_SHAP_df = pd.DataFrame()
    # Combined_SHAP_df.index.name = "ConfigLevel"

    for i in range(shapoutputsize):
        # shapoutputsize = the number of bins = number of bin configurations.
        # i::shapoutputsize 'slices' a range, it is a type of modulo operator. So when i = 0, letting shapoutputsize = 4
        # and n >= 0 we get n * shapoutputsize + i. For example, with shapoutputsize = 4 and i = 0, starting with n = 0
        # which increments by 1 we get the records: (0 * 4 + 0, 1 * 4 + 0, 2 * 4 + 0, ...,n*4+0) = (0,1*4+0,...,n*4+0).
        # If i = 2 then we get (2,6,...,n*4 + 2).
        SHAPLevel = df.iloc[i::shapoutputsize, :]

        # Each of the records associated to each bin/level e.g. (0,4,...) or (2,6,...) can be averaged together and put
        # in a dataframe.
        SHAP_Level_df = SHAPLevel.mean(axis=0).to_frame().T
        # One by one, the averaged item level SHAP values associated to each bin array are put concatentated into a dataframe.
        Combined_SHAP_df = pd.concat([Combined_SHAP_df, SHAP_Level_df])
        # Combined_SHAP_df.loc["ConfigLevel"] = SHAP_Level_df.values.tolist()
        # Conlistval = ConfigurationList[i]
        # SHAPLevelList = SHAP_Level_df.iloc[i]
        # CombinedList = Combined_SHAP_df.iloc[i]
        #
        # Combined_SHAP_df.iloc[i] = SHAP_Level_df.iloc[i]



    # Combined_SHAP_df.index.values[0] = "Level 1"
    # Combined_SHAP_df.loc[0] = "Level 1"
    # Combined_SHAP_df.set_index('id')

    # Reset the index.
    Combined_SHAP_df.index = ConfigurationList
    # Combined_SHAP_df = Combined_SHAP_df.reset_index()
    # Combined_SHAP_df.index = Combined_SHAP_df.index + 1
    # # Drop the extra column that was created when resetting the index.
    # Combined_SHAP_df = Combined_SHAP_df.drop('index', 1)
    # for i in range():
    #     ConfigurationList.append("ConfigurationLevel_" + str(i + 1))

    SHAP_Column_Names_ls = []
    # Combined_SHAP_df.index = np.arange(1, len(Combined_SHAP_df) + 1)
    # Below I'm ordering each of the bin configurations in descending order - but I don't do anything with them!
    for i in range(len(Combined_SHAP_df)):
        # Here we sort the items by the overall SHAP value and append them to a list.
        SHAP_Column_Names = Combined_SHAP_df.iloc[[i]].apply(lambda x: x.sort_values(ascending=False), axis=1)
        SHAP_Column_Names_ls.append(SHAP_Column_Names.columns.values.tolist())

    # Here, we are averaging all the SHAP values for all question configurations together.
    # That is, we average each of the SHAP values by question for all levels.
    Summed = Combined_SHAP_df.sum().to_frame().T
    print('Unordered SHAP' + str(shapoutputsize) + " Bins \n" + str(Summed))
    OrderedSHAP = Summed.apply(lambda x: x.sort_values(ascending=False), axis=1)
    OrderedSHAP.index = ["AveSHAP"]
    # print("Ordered by SHAP importance: \n" + str(Summed.apply(lambda x: x.sort_values(ascending=False), axis=1)))
    return [Combined_SHAP_df, OrderedSHAP]

def NN_FindWeights(model, SummedListofWeights):
    NN_WeightCoefs_1 = model.coefs_[0]
    # # print("weights:", NN_WeightCoefs_1)
    # # print("biases: ", clf.intercepts_)
    NNList = [abs(ele) for ele in NN_WeightCoefs_1]
    NNListSum = np.sum(NNList, 1)
    Modelweights_Rounded = np.array(['%.2f' % elem for elem in NNListSum], dtype='f')
    SummedListofWeights = np.add(SummedListofWeights, Modelweights_Rounded)
    return SummedListofWeights
