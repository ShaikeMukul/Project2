class FeatureSelection:

    def __init__(self):
        self.name = input("Enter your name: ")
        self.n_features = int(input("Please enter total number of features: "))
        self.best_subset = []
        self.best_accuracy = 0.0

    def forward_selection(self):
        print("Forward Selection.")

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

        print(f"Finished search!! The best feature subset is {self.best_subset}, which has an accuracy of {self.best_accuracy}%")

if __name__ == "__main__":
    fs = FeatureSelection()
    fs.run()
