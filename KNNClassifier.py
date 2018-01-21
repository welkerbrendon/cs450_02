class KNNClassifier:
    data = []
    targets = []
    k = 1
    def __init__(self, n_neighbors):
        self.k = n_neighbors
    def fit(self, data_train, data_target):
        self.data = data_train
        self.targets = data_target
        return self
    def predict(self, data_test, possible_targets):
        all_results = []
        for test_line in data_test:
            k_results = []
            for data_line,target in zip(self.data,self.targets):
                distance = ((test_line[0] - data_line[0])**2) + ((test_line[1] - data_line[1])**2) + ((test_line[2] - data_line[2])**2) + ((test_line[3] - data_line[3])**2)
                if(len(k_results) < self.k):
                    k_results.append([distance, target])
                else:
                    i = 0
                    for item in k_results:
                        if(item[0] > distance):
                            k_results[i] = [distance, target]
                        i = i + 1
            i = 0
            closest_neighbors = []
            while(i < self.k):
                closest_neighbors.append(k_results[i][1])

            prediction = [-1]
            for num in possible_targets:
                count = k_results.count(num)
                if(count > prediction[0]):
                    prediction[0] = count
                    prediction[1] = num

            all_results.append(prediction[1])

        return all_results
