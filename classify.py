import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
## svm
from sklearn.svm import SVC


def classify(X_train, y_train, X_test, y_test, search_type='grid'):
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
    clf = SVC(kernel='linear')

    # # Define the parameters to search
    param_dist = {
       'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear'],
        'degree': [1, 2, 3, 4, 5],
        'tol': [1e-3, 1e-4, 1e-5, 1e-6]
    }

    # Perform the search
    if search_type == 'grid':
        search = GridSearchCV(clf, param_grid=param_dist, cv=5, n_jobs=-1)
    else:
        search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=5, n_iter=50)



    # Fit the model
    search.fit(X_train, y_train)
    print(search)

    ## print the best parameters
    print("Best parameters set found on development set:")
    print(search.get_params)
    print(search.best_params_)

    # Predict the test data
    y_pred = search.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Average Accuracy: ", accuracy)

    # Print the classification report
    print(classification_report(y_test, y_pred))

    return accuracy



if __name__ == '__main__':
    # Load the data
    X_train = np.load('train_data_array_features.npy')
    y_train = np.load('train_data_array_labels.npy')
    X_test = np.load('test_data_array_features.npy')
    y_test = np.load('test_data_array_labels.npy')

    ## squeeze the dimensions 1 from the features 
    X_train = np.squeeze(X_train, axis=1)
    X_test = np.squeeze(X_test, axis=1)


    print(X_train.shape)
    # # Classify the data
    accuracy = classify(X_train, y_train, X_test, y_test)
    # print('Accuracy:', accuracy)