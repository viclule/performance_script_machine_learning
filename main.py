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

from models.runner import (execute_model_and_log,
                           generate_empty_dataframe,
                           save_df_to_csv)
from models_config import tests


if __name__ == '__main__':
    
    processor_capacity = sys.argv[1]
    logger.info(
        'Script is starting with CPU at {} %.'.format(processor_capacity))

    # create empty dataframe to collect the results
    df = generate_empty_dataframe()

    # run the tests
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
        logger.info('Polynomial took {} miliseconds to complete.'.format(time))

    # save to csv
    save_df_to_csv(df, processor_capacity)
