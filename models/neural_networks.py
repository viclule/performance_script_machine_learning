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

    def build_model(self, layers=(20,20,0,0)):
        """
        Build and train a model with the provided characteristics.
        """
        # build the base model

        self.pipeline = \
            Pipeline([('scl', StandardScaler()),
                      ('poly_features', PolynomialFeatures(degree=self.degree)),
                      ('estimator', KerasRegressor(
                        build_fn=self._base_model(layers=layers),
                        epochs=1, batch_size=50, verbose=0))])
        self._train_model()

    def _base_model(self, layers=(20,20,0,0)):
        assert layers[0] > 0
        def bm():
            model = Sequential()
            model.add(Dense(layers[0],
                            input_dim=self.get_number_of_features_combinations(),
                            kernel_initializer='normal', activation='relu'))
            if layers[1] != 0:
                model.add(Dense(layers[1], kernel_initializer='normal',
                                activation='relu'))
            if layers[2] != 0 is not None:
                model.add(Dense(layers[2], kernel_initializer='normal',
                                activation='relu'))
            if layers[3] != 0 is not None:
                model.add(Dense(layers[3], kernel_initializer='normal',
                                activation='relu'))
            model.add(Dense(1, kernel_initializer='normal', activation='linear'))
            model.compile(loss='mean_squared_error', optimizer='adam',
                        metrics=['mean_squared_error'])
            return model
        return bm
