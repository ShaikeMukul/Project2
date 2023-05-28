import random
import numpy as np


class FeatureSelection:

    def __init__(self):
        self.name = input("Enter your name: ")
        self.n_features = 0
        self.best_subset = []
        self.best_accuracy = 0.0

    def read_file(self):
        self.labels = []
        self.values = []
        file_name = "small-test-dataset.txt"
        #file_name = "large-test-dataset.txt"
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

        for i in range(len(values)):
            vector_i = curr_vals[i]
            label_i = labels[i]

            nearest_index_of_neighbour = None
            nearest_val_of_neighbour = float('inf')

            for j in range(len(values)):
                if i == j:
                    continue
                vector_j = curr_vals[j]

                distance = np.linalg.norm(vector_i - vector_j)
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

        rand_acc = self.random_accuracy()
        print(f"Using no features and “random” evaluation, I get an accuracy of {round(rand_acc * 100, 3)}%\n")

        while len(feature_set) < self.n_features:
            best_accuracy = float('-inf')
            best_feature = None
            for feature in range(1, self.n_features + 1):
                if feature in feature_set:
                    continue

                new_accuracy = self.leave_one_out_accuracy(labels, values, feature_set, feature)
                print(f'Using feature(s) {feature_set.union({feature})} accuracy is {round(new_accuracy * 100, 3)}%')

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

        rand_acc = self.random_accuracy()
        print(f"Using no features and 'random' evaluation, I get an accuracy of {round(rand_acc * 100, 3)}%")

        while len(feature_set) > 1:
            best_accuracy = float('-inf')
            best_feature = None
            for feature in range(1, self.n_features + 1):
                if feature not in feature_set:
                    continue
                updated_feature_set = feature_set.difference({feature})
                new_accuracy = self.leave_one_out_accuracy(labels, values, updated_feature_set)
                print(f'Using feature(s) {updated_feature_set}, accuracy is {round(new_accuracy * 100, 3)}%\n')

                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_feature = feature

            if best_accuracy >= best_set_accuracy:
                best_set_accuracy = best_accuracy
                best_feature_set = feature_set.difference({best_feature})

            print(f'Feature set {feature_set.difference({best_feature})} was best, accuracy is {round(best_accuracy * 100, 3)}%')
            print('\n')
            feature_set = feature_set.difference({best_feature})

            if best_set_accuracy < rand_acc:
                print("(Warning, Accuracy has decreased!)")
        
        print(f'Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {round(best_set_accuracy * 100, 3)}%')



    def special_algorithm(self):
        print(f"{self.name}’s Special Algorithm.")

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
