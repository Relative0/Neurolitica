
import random
import numpy as np


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

def syntheticdata(choice, Subjects, Questions):
    ListsofQuestions = []
    ListofVals = []
    # Dirichlet generation.
    if choice == 1:
        for x in range(Subjects):
            a, b = findweights()
            # Change a and b multipliers to change graph thickness
            [DirichletProbabilities] = np.random.dirichlet((a, 2 * a, 2 * (a ** 2 + b ** 2), 2 * b, b), size=1).round(
                10)
            AnsweredQuestion = AnswerQuestions(Questions, DirichletProbabilities)
            # ListofVals.append(DirichletProbabilities.tolist())
            ListsofQuestions.append(AnsweredQuestion.tolist())
    # Random generation.
    elif choice == 2:
        ListsofQuestions = [[random.randint(1, 5) for j in range(Questions)] for i in range(Subjects)]

    else:
        print("Issue in syntheticdata(choice) function")

    return ListsofQuestions