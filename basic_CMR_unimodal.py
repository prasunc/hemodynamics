import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, matthews_corrcoef
from kale.pipeline.mpca_trainer import MPCATrainer


# Single modality implementation
# Load Preprocessed Four Chamber or Short-axis data
x_train=np.load('fc128train.npy', allow_pickle=True)
x_test=np.load('fc128test.npy', allow_pickle=True)

# Load train and test labels 

train_labels=np.load('train_labels.npy', allow_pickle=True)
test_labels=np.load('test_labels.npy', allow_pickle=True)


x_train = np.concatenate([x_train[i].reshape((1,) + x_train[i].shape) for i in range(len(x_train))], axis=0)
x_test = np.concatenate([x_test[i].reshape((1,) + x_test[i].shape) for i in range(len(x_test))], axis=0)

#use MPCA trainer with Support Vector Machine

trainer = MPCATrainer(classifier="linear_svc",classifier_param_grid={"C": [0.00001,0.0001, 0.001, 0.01]}, n_features=210)

trainer.fit(x_train, train_labels)

y_pred = trainer.predict(x_test)
y_score = trainer.decision_function(x_test)
 
fpr, tpr, thresholds = roc_curve(test_labels, y_score)
lw=2
plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_score(test_labels, y_score))
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
