import numpy as np

class NaiveBayesScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = np.array([np.mean(y == c) for c in self.classes])
        
        self.probs = []
        for c in self.classes:
            X_c = X[y == c]
            class_probs = []
            for col in X_c.T:
                vals, counts = np.unique(col, return_counts=True)
                smoothed_probs = dict(zip(vals, (counts + 1) / (len(col) + len(vals))))
                class_probs.append(smoothed_probs)
            self.probs.append(class_probs)

    def predict(self, X):
        return np.array([self._predict_sample(s) for s in X])

    def _predict_sample(self, sample):
        posteriors = []
        for i, c in enumerate(self.classes):
            log_prob = np.log(self.priors[i])
            
            for j, val in enumerate(sample):
                prob_dict = self.probs[i][j]
                prob = prob_dict.get(val, 1 / (len(prob_dict) + 1))
                log_prob += np.log(prob)
            
            posteriors.append(log_prob)
        return self.classes[np.argmax(posteriors)]