import random


class FeatureSelection:

    def __init__(self):
        self.name = input("Enter your name: ")
        self.n_features = int(input("Please enter total number of features: "))
        self.best_subset = []
        self.best_accuracy = 0.0

    def leave_one_out_accuracy(self, feature_set, feature, labels=None, values=None):
        return random.uniform(0, 1)

    def forward_selection(self):
        feature_set = set()
        best_feature_set = None
        best_set_accuracy = float('-inf')

        while len(feature_set) < self.n_features:
            best_accuracy = float('-inf')
            best_feature = None
            for feature in range(1, self.n_features + 1):
                if feature in feature_set:
                    continue

                new_accuracy = self.leave_one_out_accuracy(feature_set, feature)
                print(f'Using feature(s) {feature_set.union({feature})} accuracy is {round(new_accuracy * 100, 3)}%')

                if new_accuracy > best_accuracy:
                    best_accuracy = new_accuracy
                    best_feature = feature

            if best_accuracy > best_set_accuracy:
                best_set_accuracy = best_accuracy
                best_feature_set = feature_set.union({best_feature})

            print(f'Feature set {feature_set.union({best_feature})} was best, accuracy is {round(best_accuracy * 100, 3)}%')
            print('\n')
            feature_set = feature_set.union({best_feature})

        print(f'Finished search!! The best feature subset is {best_feature_set}, which has an accuracy of {round(best_set_accuracy * 100, 3)}%')

    def backward_elimination(self):
        print("Backward Elimination.")

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