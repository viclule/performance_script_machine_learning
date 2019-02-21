import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures


class Model:
    """
    A base class to build dummy ML models and measure its execution time.
    """

    def __init__(self, number_of_features, degree, features_range=(0,1),
                 target_range=(0,1), number_of_training_points=100):
        self.number_of_features = number_of_features
        self.degree = degree
        self.features_range = features_range
        self.target_range = target_range
        self.number_of_training_points = number_of_training_points
        self.pipeline = None

    def _generate_dummy_data(self, number_of_datapoints):
        """
        Number of data pairs to be generated.
            :param self: 
            :param number_of_datapoints: 
        """
        # features
        train_x = np.random.rand(number_of_datapoints, self.number_of_features)
        train_x = self.features_range[0] + \
                    train_x * (self.features_range[1] - self.features_range[0])
        # target
        train_y = np.random.rand(number_of_datapoints, 1)
        train_y = self.target_range[0] + \
                    train_y * (self.target_range[1] - self.target_range[0])
        return train_x, train_y

    def execute_predictions(self, number_of_predictions):
        """
        Execute the model a number of times.
            :param self: 
            :param number_of_executions: number of predictions
        """
        test_x, _ = self._generate_dummy_data(number_of_predictions)

        # start timing
        start = time.time()
        #print(test_x.shape)
        for prediction in range(number_of_predictions):
            #print(test_x[:,prediction].shape)
            self.pipeline.predict([test_x[prediction,:]])
        return time.time() - start

    def _train_model(self):
        """
        Train the model with some dummy data.
        """
        train_x, train_y = \
            self._generate_dummy_data(self.number_of_training_points)
        self.pipeline.fit(train_x, train_y)

    def get_number_of_features_combinations(self):
        """
        Extract the number of features combinations.
        """
        poly = PolynomialFeatures(degree=self.degree)
        train_x, _ = \
            self._generate_dummy_data(self.number_of_training_points)
        poly.fit(train_x)
        return poly.n_output_features_