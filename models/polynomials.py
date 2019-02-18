from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from models.model import Model


class PolynomialModel(Model):
    """
    A class to build a polynoomial model and measure its execution time.
    """

    def build_model(self):
        """
        Build and train a model with the provided characteristics.
        """
        self.pipeline = \
            Pipeline([('scl', StandardScaler()),
                      ('poly_features', PolynomialFeatures(degree=self.degree)),
                      ('lin_regression', LinearRegression())])
        self._train_model()

 
