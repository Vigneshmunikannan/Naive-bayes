# Naive-bayes

# Algorithm
1. Convert the given dataset into frequency tables.
2. Generate Likelihood table by finding the probabilities of given features.
3. Fitting Naive Bayes to the Training set
4. Predicting the test result
5. Test accuracy of the result (Creation of Confusion matrix)
6. Visualizing the test set result.

# Code
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def train(self, X_train, y_train):
        # Calculate class probabilities
        total_samples = len(y_train)
        unique_classes = set(y_train)
        for class_label in unique_classes:
            class_count = y_train.count(class_label)
            self.class_probabilities[class_label] = class_count / total_samples

        # Calculate feature probabilities
        num_features = len(X_train[0])
        for class_label in unique_classes:
            self.feature_probabilities[class_label] = {}
            for feature_index in range(num_features):
                feature_values = [X_train[i][feature_index] for i in range(total_samples)]
                unique_feature_values = set(feature_values)
                for value in unique_feature_values:
                    feature_count = feature_values.count(value)
                    conditional_prob = feature_count / class_count
                    if feature_index not in self.feature_probabilities[class_label]:
                        self.feature_probabilities[class_label][feature_index] = {}
                    self.feature_probabilities[class_label][feature_index][value] = conditional_prob

    def predict(self, X_test):
        predictions = []
        for instance in X_test:
            instance_probs = {}
            for class_label in self.class_probabilities:
                instance_probs[class_label] = self.class_probabilities[class_label]
                for feature_index, feature_value in enumerate(instance):
                    if feature_value in self.feature_probabilities[class_label][feature_index]:
                        instance_probs[class_label] *= self.feature_probabilities[class_label][feature_index][feature_value]
                    else:
                        instance_probs[class_label] = 0  # If the feature value is not observed in training, assume zero probability

            predicted_class = max(instance_probs, key=instance_probs.get)
            predictions.append(predicted_class)

        return predictions

#Example usage:
#Sample training data
X_train = [
    [1, 'Sunny', 'Hot'],
    [2, 'Sunny', 'Hot'],
    [3, 'Overcast', 'Hot'],
    [4, 'Rain', 'Mild'],
    [5, 'Rain', 'Cool'],
    [6, 'Rain', 'Cool'],
    [7, 'Overcast', 'Cool'],
    [8, 'Sunny', 'Mild'],
    [9, 'Sunny', 'Cool'],
    [10, 'Rain', 'Mild']
]

y_train = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']

#Create and train the Naive Bayes classifier
classifier = NaiveBayesClassifier()
classifier.train(X_train, y_train)

#Sample test data
X_test = [
    [11, 'Sunny', 'Mild'],
    [12, 'Overcast', 'Hot'],
    [13, 'Rain', 'Cool'],
]

#Make predictions
predictions = classifier.predict(X_test)

#Display predictions
for i in range(len(X_test)):
    print(f"Instance {X_test[i]} is predicted as: {predictions[i]}")


# Link to run if this copy paste is not working
https://replit.com/@vigneshm2021csb/naive-bayes#main.py


# input
X_test = [
    [11, 'Sunny', 'Mild'],
    [12, 'Overcast', 'Hot'],
    [13, 'Rain', 'Cool'],
]

# output
Instance [11, 'Sunny', 'Mild'] is predicted as: No
Instance [12, 'Overcast', 'Hot'] is predicted as: Yes
Instance [13, 'Rain', 'Cool'] is predicted as: Yes




