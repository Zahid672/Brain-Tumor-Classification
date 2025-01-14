import os 

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

## svm
from sklearn.svm import SVC
import pandas as pd


model_list = ['resnet50', 'resnet101', 'densenet121', 'densenet169', 'vgg16', 'vgg19', 'alexnet', 
              'resnext50_32x4d', 'resnext101_32x8d', 'shufflenet_v2_x1_0', 'mobilenet_v2', 'mnasnet0_5']
              


ML_CLASSIFIER = ['MLP', 'GaussianNB', "Adaboost", "KNN", "RFClassifier", "SVM_linear", "SVM_sigmoid", "SVM_RBF",] # "ELM"

def classify(X_train, y_train, X_test, y_test, classifier, search_type='grid'):
    """
    Classify the data using a Random Forest Classifier
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Testing data
    :param y_test: Testing labels
    :param search_type: Type of search to perform
    :return: Accuracy of the classifier
    """
    # Create a Random Forest Classifier
    #clf = RandomForestClassifier()

    if classifier == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(100,100, 50), max_iter=1000)
    elif classifier == "GaussianNB":
        clf = GaussianNB()
    elif classifier == "Adaboost":
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    elif classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier == "RFClassifier":
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    elif classifier == "SVM_linear":
        clf = SVC(kernel='linear')
    elif classifier == "SVM_sigmoid":
        clf = SVC(kernel='sigmoid')
    elif classifier == "SVM_RBF":
        clf = SVC(kernel='rbf')
    else:
        print("Invalid classifier")
        return None
    
   # clf = SVC(kernel='linear')

    # # # Define the parameters to search
    # param_dist = {
    #    'C': [0.1, 1, 10, 100, 1000],
    #     'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #     'kernel': ['linear'],
    #     'degree': [1, 2, 3, 4, 5],
    #     'tol': [1e-3, 1e-4, 1e-5, 1e-6]
    # }

    # # Perform the search
    # if search_type == 'grid':
    #     search = GridSearchCV(clf, param_grid=param_dist, cv=5, n_jobs=-1)
    # else:
    #     search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=5, n_iter=50)



    # Fit the model
    clf.fit(X_train, y_train)

    # Predict the test data
    y_pred = clf.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy



if __name__ == '__main__':

    columns = ['Model', 'MLP', 'GaussianNB', "Adaboost", "KNN", "RFClassifier", "SVM_linear", "SVM_sigmoid", "SVM_RBF"]

    dataframe = pd.DataFrame(columns=columns)
    ## add 12 rows to the dataframe with zero values 
    for model in model_list:
        new_row = {'Model': model, 'MLP': 0, 'GaussianNB': 0, "Adaboost": 0, "KNN": 0, "RFClassifier": 0, "SVM_linear": 0, "SVM_sigmoid": 0, "SVM_RBF": 0} 
        dataframe.loc[len(dataframe)] = new_row

    main_path = 'extracted_features_1'
    for ml_classifier in ML_CLASSIFIER:
        for model in model_list:
            print('Model:', model)

            sub_dir = os.path.join(main_path, model)
            # Load the data
            X_train = np.load(os.path.join(sub_dir, 'train_data_array_features.npy'))
            y_train = np.load(os.path.join(sub_dir, 'train_data_array_labels.npy'))
            X_test = np.load(os.path.join(sub_dir, 'test_data_array_features.npy'))
            y_test = np.load(os.path.join(sub_dir, 'test_data_array_labels.npy'))


            ## squeeze the dimensions 1 from the features 
            X_train = np.squeeze(X_train, axis=1)
            X_test = np.squeeze(X_test, axis=1)

            # # Classify the data
            accuracy = classify(X_train, y_train, X_test, y_test, ml_classifier)
            print('Accuracy:', accuracy)

            dataframe.loc[dataframe['Model'] == model, ml_classifier] = accuracy

        print(dataframe)
        dataframe.to_csv('BT-large-2c-dataset_results.csv', index=False)
