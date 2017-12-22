import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from ensemble import AdaBoostClassifier
import numpy as np


class Hierarchy_Adaboost:
    '''an simple adaboost variant for solving multi-class classification problems'''

    def __init__(self, weakers_limit=10):
        # initialize adaboost classifier container
        self.adaboosts = []
        # set the maximum number of weaker calssifiers
        self.maximum_weakers = weakers_limit

    # seperate face and nonface labels
    def seperate_face_nonface(y):
        # initialize the container
        labels = []
        # judge one by one
        for i in range(len(y)):
            # if y=1 or 2,then label it as 1.otherwise label it as -1
            if y.iat[i] == 1 or y.iat[i] == 2:
                labels.append(1)
            else:
                labels.append(-1)

        return labels

    # extract data of male and female images
    def extract_male_female(X, y):
        # initialize the container
        features = []
        labels = []
        # extract data one by one
        for i in range(len(y)):
            # if y=1,then it is images of males and label it as 1.if y=2,then it is images of females and label it as -1.
            if y.iat[i] == 1:
                features.append(X[i])
                labels.append(1)
            elif y.iat[i] == 2:
                features.append(X[i])
                labels.append(-1)
            else:
                continue

        return features, labels

    # extract animals and objects
    def extract_animal_object(X, y):
        # initialize the container
        features = []
        labels = []
        # extract data one by one
        for i in range(len(y)):
            # if y=3,then it is images of males and label it as 1.if y=4,then it is images of females and label it as -1.
            if y.iat[i] == 3:
                features.append(X[i])
                labels.append(1)
            elif y.iat[i] == 4:
                features.append(X[i])
                labels.append(-1)
            else:
                continue

        return features, labels

    # train the hierarchy adaboost model
    def fit(self, X, y):
        # clear the adaboost container
        self.adaboosts.clear()
        # data preprocess
        y_face_nonface = Hierarchy_Adaboost.seperate_face_nonface(y)
        X_male_female, y_male_female = Hierarchy_Adaboost.extract_male_female(X, y)
        X_animal_object, y_animal_object = Hierarchy_Adaboost.extract_animal_object(X, y)
        # initialize a decision tree classifier
        dt = DecisionTreeClassifier(max_depth=4)
        # train an adaboost for each different situation
        # adaboost for classifying face images and nonface images
        print("train adaboost_face_nonface")
        adaboost_face_nonface = AdaBoostClassifier(dt, self.maximum_weakers)
        adaboost_face_nonface.fit(X, y_face_nonface)
        self.adaboosts.append(adaboost_face_nonface)
        # adaboost for classifying male images and female images
        print("train adaboost_male_female")
        adaboost_male_female = AdaBoostClassifier(dt, self.maximum_weakers)
        adaboost_male_female.fit(X_male_female, y_male_female)
        self.adaboosts.append(adaboost_male_female)
        # adaboost for classifying animal images and object images
        print("train adaboost_animal_object")
        adaboost_animal_object = AdaBoostClassifier(dt, self.maximum_weakers)
        adaboost_animal_object.fit(X_animal_object, y_animal_object)
        self.adaboosts.append(adaboost_animal_object)

    # predict function
    def predict(self, X):
        # get the prediction of face or nonface
        pred_face_nonface = self.adaboosts[0].predict(X)
        # initialize the container
        pred = np.zeros_like(pred_face_nonface)
        # deal with the second hirarchy
        for i in range(len(pred_face_nonface)):
            # use corresponding adaboost to classify male or female
            if pred_face_nonface[i] == 1:
                # change the form of x
                x_sample = X[i].reshape(1, -1)
                y_sample = self.adaboosts[1].predict(x_sample)
                # if y_sample=1,then label the sample as 1. if y_sample=-1,then label the sample as 2
                if y_sample == 1:
                    pred[i] = 1
                else:
                    pred[i] = 2
            else:
                # use corresponding adaboost to classify animal or object
                # change the form of x
                x_sample = X[i].reshape(1, -1)
                y_sample = self.adaboosts[2].predict(x_sample)
                # if y_sample=1,then label the sample as 3. if y_sample=-1,then label the sample as 4
                if y_sample == 1:
                    pred[i] = 3
                else:
                    pred[i] = 4

        # return the result
        return pred

    # calculate the score
    def score(self, X, y):
        # get the prediction
        y_pred = Hierarchy_Adaboost.predict(self, X)
        # get the volume of data
        data_volume = len(y)
        # initialize the counter
        count = 0
        # count the correct prediction
        for i in range(len(y)):
            if y_pred[i] == y.iat[i]:
                count = count + 1

        # calculate the correct rate
        correct_rate = count / data_volume
        return correct_rate