# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, matthews_corrcoef
from kale.pipeline.mpca_trainer import MPCATrainer

# Implementation of hybrid trimodal for PAWP prediction


# Load Preprocessed Four Chamber data
x_train1=np.load('fc128train.npy', allow_pickle=True)
x_test1=np.load('fc128test.npy', allow_pickle=True)

# Load Preprocessed Short-axis data
x_train2=np.load('sx128train.npy', allow_pickle=True)
x_test2=np.load('sx128test.npy', allow_pickle=True)


# Perform early fusion using FC and SX
x_train=np.concatenate([x_train1,x_train2],axis=1)
x_test=np.concatenate([x_test1,x_test2],axis=1)



# Load train and test labels 

train_labels=np.load('train_labels.npy', allow_pickle=True)
test_labels=np.load('test_labels.npy', allow_pickle=True)


x_train = np.concatenate([x_train[i].reshape((1,) + x_train[i].shape) for i in range(len(x_train))], axis=0)
x_test = np.concatenate([x_test[i].reshape((1,) + x_test[i].shape) for i in range(len(x_test))], axis=0)

# train the models for early fusion


trainer = MPCATrainer(classifier="linear_svc",classifier_param_grid={"C": [0.00001,0.0001, 0.001, 0.01]}, n_features=210)


trainer.fit(x_train, train_labels)


y_pred = trainer.predict(x_test)
y_score = trainer.decision_function(x_test)

# Get the baseline scores for Cardiac Measurements (CM)
measurements=pd.read_csv('baseline.csv',index_col='patient_id')
# Extract baseline scores

base_score=measurements['score'].values

base_score = (base_score-np.min(base_score))/(np.max(base_score)-np.min(base_score))

y_score = (y_score-np.min(y_score))/(np.max(y_score)-np.min(y_score))

# late fusion of decision scores

final_score=0.5*base_score+0.5*y_score

#thresholding the final score

y_pred=np.zeros(len(y_pred1))

for i in range(len(final_score)):
    if final_score[i]>=0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0
        

#plotting the ROC curve

 fpr, tpr, thresholds = roc_curve(test_labels, final_score)
lw=2
plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(test_labels, final_score))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([-0.05, 1.05])
plt.show()

print("Accuracy- {:.4f}" .format(accuracy_score(test_labels, y_pred)))
print("AUC- {:.4f}" .format(roc_auc_score(test_labels, y_score)))
print("MCC Score-{:.4f}" .format(matthews_corrcoef(test_labels, y_pred)))



