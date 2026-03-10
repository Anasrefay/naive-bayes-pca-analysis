import numpy as np

class GaussianNBScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.stats = [] 
        self.priors = np.array([np.mean(y == c) for c in self.classes])

        for c in self.classes:
            X_c = X[y == c]
            self.stats.append([(np.mean(col), np.var(col) + 1e-9) for col in X_c.T])

    def predict(self, X):
        return np.array([self._predict_sample(s) for s in X])

    def _predict_sample(self, sample):
        posteriors = []
        for i, c in enumerate(self.classes):
            log_prob = np.log(self.priors[i])
            for j, x in enumerate(sample):
                mean, var = self.stats[i][j]
                exponent = np.exp(-((x - mean)**2 / (2 * var)))
                likelihood = (1 / np.sqrt(2 * np.pi * var)) * exponent
                log_prob += np.log(likelihood + 1e-9)
            posteriors.append(log_prob)
        return self.classes[np.argmax(posteriors)]