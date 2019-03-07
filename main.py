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

from runner import (execute_model_and_log,
                    execute_fft_and_log,
                    execute_optimization_and_log,
                    generate_empty_dataframe,
                    save_df_to_csv)
from models_config import tests, tests_fft, tests_pso


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        processor_capacity = sys.argv[1]
    else:
        processor_capacity = 100.0
    logger.info(
        'Script is starting with CPU at {} %.'.format(processor_capacity))

    # create empty dataframe to collect the results
    df = generate_empty_dataframe()

    # run the model tests
    for _, test in tests.items():
        df, time = execute_model_and_log(df,
                                        test['type'],
                                        test['degree'],
                                        test['number_of_features'],
                                        test['number_of_predictions'],
                                        layers=test['layers'])
        # log it
        logger.info('Model {}, degree {} with {} features executed {} times.'.format(
                    test['type'], test['degree'], test['number_of_features'],
                    test['number_of_predictions']))
        logger.info('Model took {} miliseconds to complete.'.format(time))

    # run the fft tests
    for _, test in tests_fft.items():
        df, time = execute_fft_and_log(df,
                                        test['file_name'],
                                        test['number_of_executions'])
        # log it
        logger.info('FFT ran on file {} executed {} times.'.format(
                    test['file_name'], test['number_of_executions']))
        logger.info('FFT ran on file {} took {} miliseconds to complete.'.format(
                    test['file_name'], time))
    
    # run the optimization tests
    for _, test in tests_pso.items():
        df, time = execute_optimization_and_log(df,
                                                test['type'],
                                                test['number_of_dimensions'],
                                                test['number_of_iterations'],
                                                test['number_of_executions'],)
        # log it
        logger.info('Optimizer type {} executed {} times.'.format(
                    test['type'], test['number_of_executions']))
        logger.info('Optimizer type {} took {} miliseconds to complete.'.format(
                    test['type'], time))

    # save to csv
    save_df_to_csv(df, processor_capacity)
