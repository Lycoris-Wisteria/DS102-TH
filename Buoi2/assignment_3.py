import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score 
from sklearn.linear_model import LogisticRegression

from assignment_1 import train_set as train_set_1, test_set as test_set_1
from assignment_2 import train_set as train_set_2, test_set as test_set_2

train_images_1, train_labels_1 = train_set_1
test_images_1, test_labels_1 = test_set_1

train_images_2, train_labels_2 = train_set_2
test_images_2, test_labels_2 = test_set_2

def print_metrics(y_true, y_pred, average):
    print("Precision:", precision_score(y_true, y_pred, average=average))
    print("Recall:", recall_score(y_true, y_pred, average=average))
    print("F1 score:", f1_score(y_true, y_pred, average=average))

#LogisticRegression

logistic = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

logistic.fit(train_images_1, train_labels_1)

y_pred_1 = logistic.predict(test_images_1)
print("Logistic Regression:")
print_metrics(test_labels_1, y_pred_1, average="binary")

#SoftmaxRegression
softmax = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

# Đổi nhãn one-hot sang số lớp để sklearn sử dụng
y_train_2 = np.argmax(train_labels_2, axis=1)
y_test_2 = np.argmax(test_labels_2, axis=1)

softmax.fit(train_images_2, y_train_2)

y_pred_2 = softmax.predict(test_images_2)
print("\nSoftmax Regression:")
print_metrics(y_test_2, y_pred_2, average="macro")