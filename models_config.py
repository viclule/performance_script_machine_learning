# Models configuration

tests = {
        1 : {
            'type': 'poly',
            'degree': 2,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,0,0)
        },
        2 : {
            'type': 'poly',
            'degree': 3,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,0,0)
        },
        3 : {
            'type': 'nn',
            'degree': 1,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,0,0)
        },
        4 : {
            'type': 'nn',
            'degree': 2,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,0,0)
        },
        5 : {
            'type': 'nn',
            'degree': 1,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,20,0)
        },
        6 : {
            'type': 'nn',
            'degree': 2,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,20,0)
        },
        7 : {
            'type': 'nn',
            'degree': 1,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,20,20)
        },
        8 : {
            'type': 'nn',
            'degree': 2,
            'number_of_features': 10,
            'number_of_predictions': 1000,
            'layers': (20,20,20,20)
        },
    }


tests_fft = {
        1 : {
            'type': 'fft',
            'file_name': 'short.wav',
            'number_of_executions': 100
        },
        2 : {
            'type': 'fft',
            'file_name': 'medium.wav',
            'number_of_executions': 100
        },
        3 : {
            'type': 'fft',
            'file_name': 'long.wav',
            'number_of_executions': 1
        }
    }


tests_pso = {
        1 : {
            'type': 'pso',
            'number_of_dimensions': 3,
            'number_of_iterations': 50,
            'number_of_executions': 10
        },
        2 : {
            'type': 'pso',
            'number_of_dimensions': 4,
            'number_of_iterations': 50,
            'number_of_executions': 10
        },
        3 : {
            'type': 'pso',
            'number_of_dimensions': 5,
            'number_of_iterations': 50,
            'number_of_executions': 10
        }
    }