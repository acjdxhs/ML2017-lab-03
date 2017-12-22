import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        # initialize weak classifiers and importance scores container
        self.classifiers = []
        self.alphas = []
        # put an original classifier in classifiers container
        self.classifiers.append(weak_classifier)
        # set the maximum number of weak classifiers
        self.weakers_maximum = n_weakers_limit
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        # initialize sample weight vector
        sample_weight = np.ones_like(y) / len(y)
        # train a base classifier
        self.classifiers[-1].fit(X, y, sample_weight=sample_weight)
        # initialize accuracy container
        accuracys = []
        # train more classifiers
        for i in range(self.weakers_maximum - 1):
            # calculate the error rate
            error_rate = 1 - self.classifiers[-1].score(X, y, sample_weight=sample_weight)
            print("Weaker " + str(i + 1) + "  error rate: " + str(error_rate))
            # deal with the situation when error rate is 0
            if error_rate == 0:
                # set the importance score of the perfect classifier into 3.45(log(999)/2)
                self.alphas.append(3.45)
                break
            # calculate importance score of the classifier
            alpha = np.log((1 - error_rate) / error_rate) / 2
            # judge whether the new classifier is useful or not
            if alpha < 0:
                break
            # put the importance score into container
            self.alphas.append(alpha)
            # calculate the accuracy of the whole model
            accuracy = AdaBoostClassifier.score(self, X, y)
            print("epoch: " + str(i) + " accuracy: " + str(accuracy))
            # put new accuracy into the container
            accuracys.append(accuracy)
            # update the sample weight
            sample_weight = AdaBoostClassifier.update_sample_weight(self, X, y, sample_weight)
            # create a new classifier and train it
            weaker = DecisionTreeClassifier(max_depth=4)
            weaker.fit(X, y, sample_weight=sample_weight)
            # weaker=self.classifiers[-1]
            # weaker.fit(X,y,sample_weight=sample_weight)
            # put the new classifier into the container
            self.classifiers.append(weaker)

        # draw the accuracy graph
        x_accuracy = np.arange(len(accuracys))
        plt.plot(x_accuracy, accuracys, 'b')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
        # remove the last classifier which does not have an importance score
        if (len(self.classifiers) > len(self.alphas)):
            self.classifiers.pop()
        pass

    # calculate new sample weight and return
    def update_sample_weight(self, X, y, old_sample_weight):
        # initialize new sample weight
        new_sample_weight = old_sample_weight.copy()
        # get the importance score of the lastest classifier
        alpha = self.alphas[-1]
        # calculate new sample weight before  one by one
        for i in range(len(new_sample_weight)):
            # reshape the features vector
            x_sample = X[i]
            x_sample = x_sample.reshape(1, -1)
            # get the prediction
            hx = self.classifiers[-1].predict(x_sample)
            # calculate the factor
            factor = np.exp(-alpha * hx * y[i])
            # multiply factor and old sample weight
            new_sample_weight[i] = new_sample_weight[i] * factor

        # calculate the sum
        z = np.sum(new_sample_weight)
        # get new sample weight
        new_sample_weight = new_sample_weight / z
        return new_sample_weight

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        # initialize predict scores
        pred_scores = np.zeros(len(X))
        # sum the weighted scores
        for i in range(len(self.classifiers)):
            # calculate the prediction
            y_prediction = self.classifiers[i].predict(X)
            # get the importance score
            alpha = self.alphas[i]
            # add the weighted score to predict scores
            pred_scores = pred_scores + alpha * y_prediction

        return pred_scores
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        # calculate the predict score
        pred_score = AdaBoostClassifier.predict_scores(self, X)
        # initialize predict result
        pred_result = np.zeros_like(pred_score)
        # mark samples whose predict scores are larger than threshold as positive ,on the contrary as negative
        for i in range(len(pred_score)):
            if (pred_score[i] > threshold):
                pred_result[i] = 1
            else:
                pred_result[i] = -1

        return pred_result
        pass

    # evaluation the model using test data
    def score(self, X, y):
        # get the prediction
        y_pred = AdaBoostClassifier.predict(self, X)
        # get the data volume
        data_volume = len(y)
        # initialize a counter
        counter = 0
        # count the correct prediction
        for i in range(data_volume):
            # compare the prediction and the truth
            if y[i] == y_pred[i]:
                counter = counter + 1

        return counter / data_volume

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
