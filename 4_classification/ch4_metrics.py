import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def get_metrics(X, y, clf, threshold=0.5):
    # Get metrics for binary classifier
    y_probabilities = clf.predict_proba(X)
    y_pred = (y_probabilities[:, 1] > threshold).astype(int)
    
    unique_results = np.unique(y, return_counts=True)
    most_common_label = unique_results[0][unique_results[1].argmax()]
    y_null = np.full_like(y, most_common_label)

    accuracy = accuracy_score(y, y_pred)
    null_accuracy = accuracy_score(y, y_null)
    confusion = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred).round(3)
    precision = precision_score(y, y_pred).round(3)
    recall = recall_score(y, y_pred).round(3)

    return accuracy, null_accuracy, confusion, f1, precision, recall

def display_metrics(accuracy, null_accuracy, confusion, f1, precision, recall):
    # Print metrics for binary classifier
    print('Accuracy:', accuracy)
    print('Null Accuracy:{0}\n'.format(null_accuracy))
    
    print('\nConfusion Matrix:\n{0}\n'.format(confusion))
    print('F1 Score:', f1)
    print('Precision:', precision)
    print('Recall:',  recall)

def evaluate_classifier(X, y, clf, threshold=0.5):
    # Evaluate binary classifier 
    metrics = get_metrics(X, y, clf, threshold=threshold)
    display_metrics(*metrics)
    
