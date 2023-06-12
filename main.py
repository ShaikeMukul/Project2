import random
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
# import time


class FeatureSelection:

    def __init__(self):
        self.values = None
        self.labels = None
        self.name = input("Enter your name: ")
        self.n_features = 0
        self.best_subset = []
        self.best_accuracy = 0.0

    def read_file(self):
        self.labels = []
        self.values = []
        file_name = input("Dataset name: ")
        # "small-test-dataset.txt"
        # "large-test-dataset.txt"
        with open(file_name) as f:
            contents = f.readline().split()
            while contents or (not self.values):
                label = contents[0]
                vals = np.array([float(num) for num in contents[1:]])
                self.labels.append(label)
                self.values.append(vals)
                contents = f.readline().split()
        # print(values[0])
        return self.labels, self.values

    def random_accuracy(self):
        return random.uniform(0, 1)

    def leave_one_out_accuracy(self, labels, values, features, feature_to_add=None):
        correct_num = 0
        curr_vals = []
        for row in values:
            curr_row = []
            for ind in range(len(row)):
                if (ind + 1 in features) or (ind + 1) == feature_to_add:
                    curr_row.append(row[ind])
            curr_vals.append(np.array(curr_row))
            
        standard_deviation = np.linalg.norm(np.std(curr_vals, axis=0))
        ##print(f'Standard deviation is: " {standard_deviation}')

        for i in range(len(values)):
            vector_i = curr_vals[i]
            label_i = labels[i]

            nearest_index_of_neighbour = None
            nearest_val_of_neighbour = float('inf')

            for j in range(len(values)):
                if i == j:
                    continue
                vector_j = curr_vals[j]

                distance = np.linalg.norm(vector_i - vector_j) / standard_deviation
                if distance < nearest_val_of_neighbour:
                    nearest_index_of_neighbour = j
                    nearest_val_of_neighbour = distance

            predict_label = labels[nearest_index_of_neighbour]
            if predict_label == label_i:
                correct_num += 1

        return correct_num / len(values)

    def forward_selection(self):
        labels, values = self.read_file()
        self.n_features = len(values[0])
        feature_set = set()
        best_feature_set = None
        best_set_accuracy = float('-inf')

        print(
            f'This dataset has {self.n_features} features (not including the class attribute), with {len(self.values)} instances.\n')

        rand_acc = self.random_accuracy()
        print(f"Running nearest neighbor with no features (default rate), using “leaving-one-out” evaluation, "
              f"I get an accuracy of {round(rand_acc * 100, 3)}% \n")

        while len(feature_set) < self.n_features:
            best_accuracy = float('-inf')
            best_feature = None
            for feature in range(1, self.n_features + 1):
                if feature in feature_set:
                    continue

                # start_time = time.time()
                new_accuracy = self.leave_one_out_accuracy(labels, values, feature_set, feature)
                print(f'Using feature(s) {feature_set.union({feature})} accuracy is {round(new_accuracy * 100, 3)}%')
                """
                elapsed_time = round(time.time() - start_time, 5)
                print(f'Time taken: {elapsed_time} seconds\n')
                """
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_feature = feature

            if best_accuracy > best_set_accuracy:
                best_set_accuracy = best_accuracy
                best_feature_set = feature_set.union({best_feature})

            print(
                f'Feature set {feature_set.union({best_feature})} was best, accuracy is {round(best_accuracy * 100, 3)}%')
            print('\n')
            feature_set = feature_set.union({best_feature})

        if best_accuracy < rand_acc:
            print("(Warning, Accuracy has decreased!)")

        print(
            f'Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {round(best_set_accuracy * 100, 3)}%')

    # this is pretty much like forward selection except for a few lines of difference
    # just change all the unions to differences, since in this one we're taking away from the feature set
    def backward_elimination(self):
        labels, values = self.read_file()
        self.n_features = len(values[0])
        feature_set = set(range(1, self.n_features + 1))  # Start with all features
        best_feature_set = feature_set
        best_set_accuracy = self.leave_one_out_accuracy(labels, values, feature_set)

        print(
            f'This dataset has {self.n_features} features (not including the class attribute), with {len(self.values)} instances.\n')

        rand_acc = self.random_accuracy()
        print(f"Running nearest neighbor with no features (default rate), using “leaving-one-out” evaluation, "
              f"I get an accuracy of {round(rand_acc * 100, 3)}% \n")

        while len(feature_set) > 1:
            best_accuracy = float('-inf')
            best_feature = None
            for feature in range(1, self.n_features + 1):
                if feature not in feature_set:
                    continue
                updated_feature_set = feature_set.difference({feature})

                # start_time = time.time()
                new_accuracy = self.leave_one_out_accuracy(labels, values, updated_feature_set)
                print(f'Using feature(s) {updated_feature_set}, accuracy is {round(new_accuracy * 100, 3)}%')
                """
                elapsed_time = round(time.time() - start_time, 5)
                print(f'Time taken: {elapsed_time} seconds\n')
                """
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_feature = feature

            if best_accuracy >= best_set_accuracy:
                best_set_accuracy = best_accuracy
                best_feature_set = feature_set.difference({best_feature})

            print(
                f'Feature set {feature_set.difference({best_feature})} was best, accuracy is {round(best_accuracy * 100, 3)}%')
            print('\n')
            feature_set = feature_set.difference({best_feature})

            if best_set_accuracy < rand_acc:
                print("(Warning, Accuracy has decreased!)")

        print(
            f'Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {round(best_set_accuracy * 100, 3)}%')


    def k_fold_cross_validation(self, labels, values, features, feature_to_add=None):
        curr_vals = []
        for row in values:
            curr_row = []
            for ind in range(len(row)):
                if (ind + 1 in features) or (ind + 1) == feature_to_add:
                    curr_row.append(row[ind])
            curr_vals.append(np.array(curr_row))

        '''
        Logistic Regression did not produce very insightful results and decision tree classifier 
        wasn't working properly so I went with RandomForestClassifier which seems to be a little more accurate.
        RandomForestClassifier works with decision trees though to make predictions.
        '''
        model = RandomForestClassifier()
        '''
        n_splits tells us how much to fold the data, from what I read, 
        5 and 10 is popular and I thought 5 would make sense for both the small and the large datasets

        shuffle=True, I read that this prevents bias and I just set it to true

        random_state=10, just ensures we get the same output every time we fold, 
        the 10 is kind of arbitrary I just picked something

        KFold function folds the data and returns indeces to each of the folds.
        '''
        k_fold = KFold(n_splits=5, shuffle=True, random_state=10)

        # returns an array of scores that correspond to the accuracy of each fold
        scores = cross_val_score(model, np.array(curr_vals), labels, cv=k_fold)
        return scores.mean()
    

# exactly the same as forward selection but it uses the k_fold_cross_validation accuracy instead
    def special_algorithm(self):
        labels, values = self.read_file()
        self.n_features = len(values[0])
        feature_set = set()
        best_feature_set = None
        best_set_accuracy = float('-inf')

        print(
            f'This dataset has {self.n_features} features (not including the class attribute), with {len(self.values)} instances.\n')

        rand_acc = self.random_accuracy()
        print(f"Running nearest neighbor with no features (default rate), using “k-fold cross validation” evaluation, "
              f"I get an accuracy of {round(rand_acc * 100, 3)}% \n")

        while len(feature_set) < self.n_features:
            best_accuracy = float('-inf')
            best_feature = None
            for feature in range(1, self.n_features + 1):
                if feature in feature_set:
                    continue

                # start_time = time.time()
                new_accuracy = self.k_fold_cross_validation(labels, values, feature_set, feature)
                print(f'Using feature(s) {feature_set.union({feature})} accuracy is {round(new_accuracy * 100, 3)}%')
                """
                elapsed_time = round(time.time() - start_time, 5)
                print(f'Time taken: {elapsed_time} seconds\n')
                """
                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_feature = feature

            if best_accuracy > best_set_accuracy:
                best_set_accuracy = best_accuracy
                best_feature_set = feature_set.union({best_feature})

            print(
                f'Feature set {feature_set.union({best_feature})} was best, accuracy is {round(best_accuracy * 100, 3)}%')
            print('\n')
            feature_set = feature_set.union({best_feature})

        if best_accuracy < rand_acc:
            print("(Warning, Accuracy has decreased!)")

        print(
            f'Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {round(best_set_accuracy * 100, 3)}%')


    def run(self):
        print(f"Welcome to {self.name}'s Feature Selection Algorithm.")
        print("Type the number of the algorithm you want to run.")
        print(f"1. Forward Selection\n2. Backward Elimination\n3. {self.name}’s Special Algorithm.")
        algorithm = int(input())

        if algorithm == 1:
            self.forward_selection()
        elif algorithm == 2:
            self.backward_elimination()
        elif algorithm == 3:
            self.special_algorithm()
        else:
            print("Invalid algorithm selection.")


if __name__ == "__main__":
    fs = FeatureSelection()
    fs.run()
