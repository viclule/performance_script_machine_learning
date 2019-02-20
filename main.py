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

from models.runner import (execution_time_polynomial,
                           execution_time_neural_network,
                           execute_model_and_log,
                           generate_empty_dataframe,
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

    df, time = execute_model_and_log(df,
                                    'poly',
                                    degree,
                                    number_of_features,
                                    number_of_predictions)
    # logging
    logger.info('Polynomial {} with {} features executed {} times.'.format(
                degree, number_of_features, number_of_predictions))
    logger.info('Polynomial took {} miliseconds to complete.'.format(time))



    # Polynomial degree 3 model execution time
    degree = 3
    number_of_features = 10
    number_of_predictions = 1000

    df, time = execute_model_and_log(df,
                                    'poly',
                                    degree,
                                    number_of_features,
                                    number_of_predictions)
    # logging
    logger.info('Polynomial {} with {} features executed {} times.'.format(
                degree, number_of_features, number_of_predictions))
    logger.info('Polynomial took {} miliseconds to complete.'.format(time))



    # NN degree 1, 1 hidden layermodel execution time
    degree = 1
    number_of_features = 10
    layers=(20,20,0,0)
    number_of_layers = len(layers) - layers.count(0) + 1
    number_of_predictions = 1000

    df, time = execute_model_and_log(df,
                                    'nn',
                                    degree,
                                    number_of_features,
                                    number_of_predictions,
                                    layers=layers)
    # logging
    logger.info('NN {} with {} features executed {} times.'.format(
                degree, number_of_features, number_of_predictions))
    logger.info('NN took {} miliseconds to complete.'.format(time))



    # NN degree 2, 1 hidden layermodel execution time
    degree = 2
    number_of_features = 10
    layers=(20,20,0,0)
    number_of_layers = len(layers) - layers.count(0) + 1
    number_of_predictions = 1000

    df, time = execute_model_and_log(df,
                                    'nn',
                                    degree,
                                    number_of_features,
                                    number_of_predictions,
                                    layers=layers)
    # logging
    logger.info('NN {} with {} features executed {} times.'.format(
                degree, number_of_features, number_of_predictions))
    logger.info('NN took {} miliseconds to complete.'.format(time))


    # save to csv
    save_df_to_csv(df, processor_capacity)
