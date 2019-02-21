import pandas as pd
import sys, os
from time import strftime

from models.polynomials import PolynomialModel
from models.neural_networks import NeuralNetworkModel

def _execution_time_polynomial(degree, number_of_features, times,
                              features_range=(0,1),
                              target_range=(0,1),
                              number_of_training_points=100):
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

    # return time and number pf features combinations
    return execution_time, model.get_number_of_features_combinations()


def _execution_time_neural_network(degree, number_of_features, times,
                                  layers=(20,20,0,0),
                                  features_range=(0,1),
                                  target_range=(0,1),
                                  number_of_training_points=100):
    """
    Measures the time it takes for a model to be executed.
        :param degree: degree of the polynomia
        :param number_of_features: number of simulated features
        :param times: times the model gets executed

        :returns: the execution time in miliseconds
    """
    # create a model instance
    model = NeuralNetworkModel(number_of_features, degree,
                            features_range=features_range,
                            target_range=target_range,
                            number_of_training_points=number_of_training_points)
    model.build_model(layers=layers)
    # execute the model
    execution_time = model.execute_predictions(times)

    # return time and number pf features combinations
    return execution_time, model.get_number_of_features_combinations()


def execute_model_and_log(df, kind, degree, number_of_features,
                          number_of_predictions,
                          layers=(20,20,0,0)):
    if kind == 'poly':
        time, combinations = _execution_time_polynomial(degree,
                                number_of_features, number_of_predictions,
                                features_range=(0,1),
                                target_range=(0,1),
                                number_of_training_points=100)
        # insert result to dataframe
        result = {'model_architecture': 'polynomial',
                'number_of_features': number_of_features,
                'degree': degree,
                'number_of_elements': combinations,
                'number_executions': number_of_predictions,
                'execution_time': time,
                'number_of_layers': None,
                'input_layer_size': None,
                'hidden_layer_1_size': None,
                'hidden_layer_2_size': None,
                'hidden_layer_3_size': None,
                'output_layer': None,}
        df.loc[len(df)] = result

    elif kind == 'nn':
        time, combinations = _execution_time_neural_network(degree,
                                number_of_features, number_of_predictions,
                                layers=layers,
                                features_range=(0,1),
                                target_range=(0,1),
                                number_of_training_points=100)
        # insert result to dataframe
        number_of_layers = len(layers) - layers.count(0) + 1
        result = {'model_architecture': 'neural network',
                'number_of_features': number_of_features,
                'degree': degree,
                'number_of_elements': combinations,
                'number_executions': number_of_predictions,
                'execution_time': time,
                'number_of_layers': number_of_layers,
                'input_layer_size': layers[0],
                'hidden_layer_1_size': layers[1],
                'hidden_layer_2_size': layers[2],
                'hidden_layer_3_size': layers[3],
                'output_layer': 1,}
        df.loc[len(df)] = result
    return df, time
    

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