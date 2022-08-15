import pandas as pd
from pandas import DataFrame
from SyntheticMethods import syntheticdata
from SupportFunctions import ChoiceImputation

def Get_or_Create_Questionnaire():

    # Num_Questions, df = Create_Synthetic_Data(11, 3906)
    Num_Questions, df = Dutch_Dataset()
    return Num_Questions, df

def Create_Synthetic_Data(Questions=10, Subjects=2000):
    # We choose the number of questions and subjects here.
    QuestionList = []
    for i in range(Questions):
        QuestionList.append("Q_" + str(i + 1))

    syntheticDatachoice = 1  # 1 = Dirichlet number generation
    SyntheticData = syntheticdata(syntheticDatachoice, Subjects, Questions)
    df = DataFrame.from_records(SyntheticData)
    df.columns = QuestionList
    return Questions, df

def Dutch_Dataset():
    # Import the particular dataset.
    PSWQ = PSWQ_Dutch_Positive()
    # PSWQ = PSWQ_Dutch_Full()

    # If other questionnaires are added, cleaned, filtered etc. they can be concatenated here.
    Questionnaire = pd.concat([PSWQ], axis=1)
    Number_Questions = len(Questionnaire.columns)

    # Choose whether to impute or drop subjects with missing item scores (1 = impute, 0 = drop columns with missing data)
    Imputationchoice = 1
    Questionnaire = ChoiceImputation(Questionnaire, Imputationchoice)
    return Number_Questions, Questionnaire

def PSWQ_Dutch_Positive():
    data = pd.read_csv('PSWQ_Dutch.csv', sep=',')
    # Will hold question configurations from file.
    PSWQ = data.filter(like='PSWQ_', axis=1)
    # Remove reverse coded questions.
    PSWQ.drop(PSWQ.columns[[0, 2, 7, 9, 10]], axis=1, inplace=True)

    return PSWQ

def PSWQ_Dutch_Full():
    data = pd.read_csv('PSWQ_Dutch.csv', sep=',')
    # Will hold question configurations from file.
    PSWQ = data.filter(like='PSWQ_', axis=1)

    return PSWQ