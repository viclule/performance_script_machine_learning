import pandas as pd
import sys, os
from time import strftime

from models.polynomials import PolynomialModel


def execution_time_polynomial(degree, number_of_features, times,
                              features_range=(0,1),
                              target_range=(0,1),
                              number_of_training_points=5):
    """
    Measures the time it takes for a model to be executed.
        :param degree: degree of the polynomia
        :param number_of_features: number of simulated features
        :param times: times the model gets executed

        :returns: the execution time in miliseconds
    """
    # create a model instance
    model = PolynomialModel(number_of_features, degree,
                            features_range=features_range,
                            target_range=target_range,
                            number_of_training_points=number_of_training_points)
    model.build_model()
    # execute the model
    execution_time = model.execute_predictions(times)

    # get number pf features combinations
    number_of_features_combinations = \
        model.get_number_of_features_combinations()
    return execution_time, number_of_features_combinations


def execution_time_neural_network(degree, number_of_features, times,
                                features_range=(0,1),
                                target_range=(0,1),
                                number_of_training_points=5):
    """
    Measures the time it takes for a model to be executed.
        :param degree: degree of the polynomia
        :param number_of_features: number of simulated features
        :param times: times the model gets executed

        :returns: the execution time in miliseconds
    """
    # create a model instance
    model = PolynomialModel(number_of_features, degree,
                            features_range=features_range,
                            target_range=target_range,
                            number_of_training_points=number_of_training_points)
    model.build_model()
    # execute the model
    execution_time = model.execute_predictions(times)

    # get number pf features combinations
    number_of_features_combinations = \
        model.get_number_of_features_combinations()
    return execution_time, number_of_features_combinations


def _generate_columns():
    columns = ['model_architecture',
               'number_of_features',
               'degree',
               'number_of_elements',
               'number_executions',
               'execution_time',
               'number_of_layers',
               'input_layer_size',
               'hidden_layer_1_size',
               'hidden_layer_2_size',
               'hidden_layer_3_size',
               'output_layer'
               ]
    return columns


def generate_empty_dataframe():
    df = pd.DataFrame(columns=_generate_columns())
    return df


def save_df_to_csv(df, processor_capacity):
    wk_dir = os.path.abspath(os.path.dirname('__file__'))
    time_now = strftime("%Y-%m-%d %H-%M-%S")
    file_name = 'log_files/' + time_now + ' at ' + processor_capacity + \
        ' CPU capacity.csv'
    file_name = os.path.join(wk_dir, file_name)
    df.to_csv(file_name, sep=';')