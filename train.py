from scipy import misc
from feature import NPDFeature
from sklearn.model_selection import train_test_split
from Hierarchy_Adaboost import Hierarchy_Adaboost
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.image
import numpy as np
import pickle
import os
import os
from PIL import Image


# save feature data using pickle
def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


# load feature data using pickle
def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


# preprocess original image
def preprocess_image():
    os.makedirs("datasets\\features")
    # preprocess face images
    for i in range(500):
        # set the fetch address
        fetch_address = "datasets\\original\\face\\face_%03d.jpg" % i
        # read jpg file
        # im = matplotlib.image.imread(fetch_address)
        # transform to the 24*24 gray image
        # xi = transform_image(im)
        # create a NPDFeature class
        xi = Image.open(fetch_address).convert('L').resize((24,24))
        xi = np.array(xi)
        npd = NPDFeature(xi)
        # extract feature
        feature = npd.extract()
        # set the filename
        filename = "datasets\\features\\face%03d.pickle" % i
        # save the feature in pickle file
        save_data(filename, feature)

    # preprocess nonface images
    for i in range(500):
        # set the fetch address
        fetch_address = "datasets\\original\\nonface\\nonface_%03d.jpg" % i
        # read jpg file
        # im = matplotlib.image.imread(fetch_address)
        # transform to the 24*24 gray image
        # xi = transform_image(im)
        # create a NPDFeature class
        xi = Image.open(fetch_address).convert('L').resize((24, 24))
        xi = np.array(xi)
        npd = NPDFeature(xi)
        # extract feature
        feature = npd.extract()
        # set the filename
        filename = "datasets\\features\\nonface%03d.pickle" % i
        # save the feature in pickle file
        save_data(filename, feature)


# transform the original rgb image to the 24*24 gray image
def transform_image(im):
    # resize the image
    im_resize = misc.imresize(im, [24, 24])
    # transform the rgb image to the gray image
    im_resize_gray = rgb2gray(im_resize)
    return im_resize_gray


# transform rgb images to gray images
def rgb2gray(im):
    return np.dot(im[..., :3], [0.299, 0.587, 0.114])


# load features
def load_preprocessed_data():
    # initialize a container
    features = []
    labels = []
    # load features of face images
    for i in range(500):
        # set the pickle filename
        filename = "datasets\\features\\face%03d.pickle" % i
        # load the feature
        feature = load_data(filename)
        # add the feature into the container
        features.append(feature)
        # add label 1 into labels
        labels.append(1)

    # load features of nonface images
    for i in range(500):
        # set the pickle filename
        filename = "datasets\\features\\nonface%03d.pickle" % i
        # load the feature
        feature = load_data(filename)
        # add the feature into the container
        features.append(feature)
        # add label 0 into labels
        labels.append(-1)

    return features, labels


# extract data of male and female images
def extract_male_female(X, y):
    # initialize the container
    features = []
    labels = []
    # extract data one by one
    for i in range(len(y)):
        # if y=1,then it is images of males and label it as 1.if y=-1,then it is images of females and label it as -1.
        if y[i] == 1:
            features.append(X[i])
            labels.append(1)
        elif y[i] == 2:
            features.append(X[i])
            labels.append(-1)
        else:
            continue

    return features, labels


if __name__ == "__main__":
    # load the images and transform them to grayscale images
    # preprocess_image()
    # print('preprocess completed')

    #x, y = load_preprocessed_data()
    if os.path.exists("datasets/features"):
        x, y = load_preprocessed_data()
    else:
        preprocess_image()
        x, y = load_preprocessed_data()

    # load multi-class labels
    multi_class_labels = pd.read_csv('0-999.csv')
    y = multi_class_labels.iloc[:, 0]
    # divide data into train data and validation data
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.3, random_state=50)

    # initialize a weak classifier
    # dtc=DecisionTreeClassifier(max_depth=1)
    # initialize the adaboost classifier
    # adaboost=AdaBoostClassifier(dtc,30)
    # fit the classifier
    # adaboost.fit(x_train,y_train)
    # get the prediction
    # y_validation_prediction=adaboost.predict(x_validation)
    # correct_rate=adaboost.score(x_validation,y_validation)
    # correct_rate=dtc.score(x_validation,y_validation)
    # initalize the hierarchy classifier
    ha = Hierarchy_Adaboost()
    # train the hierarchy classifier
    ha.fit(x_train, y_train)
    # get the prediction
    y_validation_prediction = ha.predict(x_validation)
    # correct_rate=ha.score(x_validation,y_validation)
    # print(correct_rate)
    target_names = ['class 1', 'class 2', 'class 3', 'class 3']
    # get the report
    report = classification_report(y_validation, y_validation_prediction, target_names=target_names)
    # create a file
    fd = open("report.txt", "w")
    # write the report
    fd.write(report)
    # close the file
    fd.close()

    # use decision tree to do multi classification
    # initialize a decision tree classifier
    dtc = DecisionTreeClassifier()
    # train the model
    dtc.fit(x_train, y_train)
    sc = dtc.score(x_validation, y_validation)
    print(sc)
    # get the prediction
    y_decision_tree_prediction = dtc.predict(x_validation)
    report_dtc = classification_report(y_validation, y_decision_tree_prediction, target_names=target_names)
    # create a file
    fd = open("report_dtc.txt", "w")
    # write the report
    fd.write(report_dtc)
    # close the file
    fd.close()
