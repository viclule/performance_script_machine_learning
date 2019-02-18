from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from models.model import Model


class NeuralNetworkModel(Model):
    """
    A class to build a neural network and measure its execution time.
    """

    def build_model(self, input_layer_size, output_layer_size,
                    hidden_layer_1_size=None, hidden_layer_2_size=None,
                    hidden_layer_3_size=None):
        """
        Build and train a model with the provided characteristics.
        """
        # build the base model
        base_model = self.base_model(input_layer_size=input_layer_size,
                                     output_layer_size=output_layer_size,
                                     hidden_layer_1_size=hidden_layer_1_size,
                                     hidden_layer_2_size=hidden_layer_2_size,
                                     hidden_layer_3_size=hidden_layer_3_size)

        self.pipeline = \
            Pipeline([('scl', StandardScaler()),
                      ('poly_features', PolynomialFeatures(degree=self.degree)),
                      ('estimator', KerasRegressor(build_fn=base_model,
                        epochs=1, batch_size=5, verbose=1))])
        self._train_model()

    def base_model(self, input_layer_size, output_layer_size,
                    hidden_layer_1_size=None, hidden_layer_2_size=None,
                    hidden_layer_3_size=None):
        model = Sequential()
        model.add(Dense(input_layer_size,
                        input_dim=self.get_number_of_features_combinations(),
                        init='normal', activation='relu'))
        if hidden_layer_1_size is not None:
            model.add(Dense(hidden_layer_1_size, init='normal',
                            activation='relu'))
        if hidden_layer_2_size is not None:
            model.add(Dense(hidden_layer_2_size, init='normal',
                            activation='relu'))
        if hidden_layer_3_size is not None:
            model.add(Dense(hidden_layer_3_size, init='normal',
                            activation='relu'))
        model.add(Dense(1, init='normal', activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam',
                      metrics=['mean_squared_error'])
        return model