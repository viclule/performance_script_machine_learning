'''
The script runs a number of different models in loops, times the execution
times and saved the results to a csv file.
'''
import sys
import logging


# logging configuration
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from models.runner import (execution_time_polynomial, generate_empty_dataframe,
                           save_df_to_csv)


if __name__ == '__main__':
    
    processor_capacity = sys.argv[1]
    logger.info(
        'Script is starting with CPU at {} %.'.format(processor_capacity))

    # create empty dataframe to collect the results
    df = generate_empty_dataframe()


    # Polynomial degree 2 model execution time
    degree = 2
    number_of_features = 10
    number_of_predictions = 1000
    time, number_of_features_combinations = \
        execution_time_polynomial(degree, number_of_features,
                                  number_of_predictions)

    # logging
    logger.info('Polynomial {} with {} features executed {} times.'.format(
                degree, number_of_features, number_of_predictions))
    logger.info('Polynomial took {} miliseconds to complete.'.format(time))

    # insert result to dataframe
    result = {'model_architecture': 'polynomial',
               'number_of_features': number_of_features,
               'degree': degree,
               'number_of_elements': number_of_features_combinations,
               'number_executions': number_of_predictions,
               'execution_time': time,
               'number_of_layers': None,
               'input_layer_size': None,
               'hidden_layer_1_size': None,
               'hidden_layer_2_size': None,
               'hidden_layer_3_size': None,
               'output_layer': None,}
    df.loc[len(df)] = result


    # Polynomial degree 3 model execution time
    degree = 3
    number_of_features = 10
    number_of_predictions = 1000
    time, number_of_features_combinations = \
        execution_time_polynomial(degree, number_of_features,
                                  number_of_predictions)
    # logging
    logger.info('Polynomial {} with {} features executed {} times.'.format(
                degree, number_of_features, number_of_predictions))
    logger.info('Polynomial took {} miliseconds to complete.'.format(time))

    # insert result to dataframe
    result = {'model_architecture': 'polynomial',
               'number_of_features': number_of_features,
               'degree': degree,
               'number_of_elements': number_of_features_combinations,
               'number_executions': number_of_predictions,
               'execution_time': time,
               'number_of_layers': None,
               'input_layer_size': None,
               'hidden_layer_1_size': None,
               'hidden_layer_2_size': None,
               'hidden_layer_3_size': None,
               'output_layer': None,}
    df.loc[len(df)] = result


    # NN degree 2, 1 hidden layermodel execution time
    degree = 2
    number_of_features = 10
    number_of_predictions = 1000
    time, number_of_features_combinations = \
        execution_time_polynomial(degree, number_of_features,
                                  number_of_predictions)

    # logging
    logger.info('Polynomial {} with {} features executed {} times.'.format(
                degree, number_of_features, number_of_predictions))
    logger.info('Polynomial took {} miliseconds to complete.'.format(time))

    # insert result to dataframe
    result = {'model_architecture': 'polynomial',
               'number_of_features': number_of_features,
               'degree': degree,
               'number_of_elements': number_of_features_combinations,
               'number_executions': number_of_predictions,
               'execution_time': time,
               'number_of_layers': None,
               'input_layer_size': None,
               'hidden_layer_1_size': None,
               'hidden_layer_2_size': None,
               'hidden_layer_3_size': None,
               'output_layer': None,}
    df.loc[len(df)] = result



    # save to csv
    save_df_to_csv(df, processor_capacity)
