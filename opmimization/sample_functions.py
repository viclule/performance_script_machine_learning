# Example of functions to be optimized


def sum_of_squares(position):
    result = 1
    for i in range(position.shape[0]):
        result = result + position[i]**2
    return result