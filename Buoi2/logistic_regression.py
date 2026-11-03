import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score 

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        self.w = None
        self.lr = learning_rate
        self.epochs = epochs

    def sigmoid(self, X: np.ndarray):
        """
            X: (N, D)
        """
        logits = X @ self.w #(N,)
        logits = 1 / (1 + np.exp(-logits))

        return logits #(N,)

    def loss_fn(self, y_hat: np.ndarray,  y: np.ndarray):
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        return - ((1 - y) * np.log(1 - y_hat) + y * np.log(y_hat)).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, D = X.shape
        self.w = np.zeros(D)
        loss_value = None

        self.loss_history = []

        with tqdm(range(self.epochs)) as pb:
            for epoch in pb:
        
                y_hat = self.sigmoid(X)

                loss_value = self.loss_fn(y_hat, y)
                self.loss_history.append(loss_value)

                grad = X.T @ (y_hat - y) / N # (1, D)
                self.w = self.w - self.lr * grad

                pb.set_postfix(loss=loss_value)

        return self

    def predict(self, X: np.ndarray):
        '''
            X(N, D)
        '''
        logits = self.sigmoid(X)
        predict_y = np.where(logits >= 0.5, 1, 0)

        return predict_y

    def evaluate(self, X, y) -> dict:
        y_hat = self.predict(X)

        P = precision_score(y, y_hat)
        R = recall_score(y, y_hat)
        F1 = f1_score(y, y_hat)

        return {
            "Precision": P,
            "Recall": R,
            "F1 score": F1
        }



