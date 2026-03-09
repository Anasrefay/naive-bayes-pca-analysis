import numpy as np

class NaiveBayesScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = []
        self.class_priors = {} 

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_priors[i] = len(X_c) / len(y)
            
            self.parameters.append([])
            for col in X_c.T:
                unique_elements, counts = np.unique(col, return_counts=True)
                prob = dict(zip(unique_elements, (counts + 1) / (len(col) + len(unique_elements))))
                self.parameters[i].append(prob)

    def _calculate_likelihood(self, class_idx, sample):
        log_prob = np.log(self.class_priors[class_idx])
        
        for feature_idx, feature_val in enumerate(sample):
            prob_dict = self.parameters[class_idx][feature_idx]
            prob = prob_dict.get(feature_val, 1 / (len(prob_dict) + 1))
            log_prob += np.log(prob)
        return log_prob

    def predict(self, X):
        predictions = [self._get_prediction(sample) for sample in X]
        return np.array(predictions)

    def _get_prediction(self, sample):
        posteriors = [self._calculate_likelihood(i, sample) for i in range(len(self.classes))]
        return self.classes[np.argmax(posteriors)]