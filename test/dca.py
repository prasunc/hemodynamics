
# -*- coding: utf-8 -*-


from sklearn.metrics import confusion_matrix

def dca(y_true, y_pred, threshold_range):
    """
    Performs decision curve analysis for a binary classification problem.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities of the positive class.
        threshold_range (array-like): Range of threshold probabilities to evaluate.
        
    Returns:
        array-like: Net benefit for each threshold probability.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred >= threshold_range[:len(y_pred)]).ravel()
    p = (tp + fn) / len(y_true)
    n = (tn + fp) / len(y_true)
    tnb = (tp / (tp + fp)) - ((1 - p) / (n - tp - fp + tn))
    return tnb


#cm1 = metrics.confusion_matrix(test_labels, y_pred)


# Evaluate DCA across a range of threshold probabilities
threshold_range = np.arange(0, 1.01, 0.01)


net_benefit = dca(test_labels, y_pred, [0,1])

# Plot the results
plt.plot(threshold_range, net_benefit)
plt.xlabel('Threshold probability')
plt.ylabel('Net benefit')
plt.title('Decision curve analysis')
plt.show()
