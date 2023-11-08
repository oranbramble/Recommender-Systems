import copy
from MatrixFactoriser import MatrixFactorisation
import random
import numpy as np
import csv
import math

"""
This is the main script that runs the whole system. It requires the training data to be in matrix format, and saved
in 'matrix.csv'. This should already be the case, but if need be, the MatrixMaker file can be run to make this file.

The main method to this file is the random_search method, which performs random search with k-fold cross-validation
to find the best parameters for the Matrix Factorisation algorithm. I.e., finding the best latent factor matrices
to provide the most accurate predicted ratings. It then predicts ratings for the ones listed in 'test.csv' and
saves the results to 'test_results.csv'.
"""


def cross_validation(num_users,  num_items, folds, cycles, params):
    """
    Method to perform cross-validation in order to find the best parameters for the matrix factorisation algorithm

    :param num_users: Number of users
    :param num_items: Number of items
    :param folds: List containing folds
    :param cycles: Number of cycles in validation
    :param params: List of parameters trying to cross-validate
    :return:
    """

    fold_mfs = []
    fold_maes = []
    for fold in folds:
        train_data = copy.deepcopy(fold[0])
        test_data = copy.deepcopy(fold[1])
        print(train_data[0])
        if len(test_data) != 0:
            all_cycle_mfs = []
            all_cycle_maes = []
            for c in range(cycles):
                mf = MatrixFactorisation(train_data, num_users, num_items,
                                            params[0], params[1], params[2], params[3])
                mf.initialise()
                train_mae = mf.train()

                all_cycle_maes.append(get_mae(test_data, mf))
                all_cycle_mfs.append(mf)
                print("CYCLE {} DONE, TRAIN ERROR {}".format(c, train_mae))

            min_mae = min(all_cycle_maes)
            index_of_min = np.argmin(np.array(all_cycle_maes))
            min_mf = all_cycle_mfs[index_of_min]

            fold_mfs.append(min_mf)
            fold_maes.append(min_mae)
    average_mae, best_mae, average_mf, best_mf = get_averages_and_best(fold_maes,fold_mfs)
    return average_mae, best_mae, average_mf, best_mf


def random_search(iterations, k, cycles):
    """
    Method to perform random search to test a range of different hyperparameters in order to find the most accurate
    based on testing error. A grid is formed of potential hyperparameters, then a random set of these is selected to
    test. K-fold cross-validation is used with each run of the random search to test the randomly selected
    hyperparameters. The best hyperparameters are selected based on the MAE on the test data. After finding the optimal
    parameters, the Matrix Factorisation algorithm is then run using these

    :param iterations: Number of search runs
    :param k: Number of folds for cross validation
    :param cycles: Number of cycles
    :return: None
    """

    # Matrix Factorisation has hyperparameters:
    #   - latent_features
    #   - learning_rate
    #   - regularisation_parameter
    #   - iterations
    grid = {
        "latent_features" : [x for x in range(30, 70, 10)],
        "learning_rate" : [0.1, 0.01, 0.001, 0.0001],
        "reg_param" : [0.1, 0.01,0.001,0.0001],
        "iters" : [10]
    }

    data, num_users, num_items = read_file("train.csv")
    # Generates all the folds used for k-fold-cross-validation
    folds = generate_folds(data, k)
    best_maes = []
    average_maes = []
    best_mfs = []
    average_mfs = []
    all_random_params = []
    # Begins the random search loop
    for i in range(iterations):
        random_params = []
        for value_ranges in grid.values():
            # Samples the grid of possible hyperparameters randomly
            random_params.append(random_sample(value_ranges))
        print("Iteration {} begins!".format(i+1))
        print("Params: LF - {}, LR - {}, RP - {}, I - {}"
              .format(random_params[0], random_params[1],random_params[2],random_params[3]))
        # Call the cross validation method which performs k-fold-cross-validation with the currrent, randomly selected
        # hyperparameters
        average_mae, best_mae, average_mf, best_mf = cross_validation(num_users, num_items, folds, cycles, random_params)
        # Saves the results
        average_maes.append(average_mae)
        best_maes.append(best_mae)
        average_mfs.append(average_mf)
        best_mfs.append(best_mf)
        all_random_params.append(random_params)
        # Logs results
        print("Iteration {} complete!".format(i+1))
        print()
    # Finds best based on the smallest average testing MAE
    minimum_mae_index = np.argmin(average_maes)
    minimum_mae = average_maes[minimum_mae_index]
    minimum_mf = average_mfs[minimum_mae_index]

    print("FINISHED!!!!")
    print("PREDICTING RESULTS TO FILE NOW")
    # Makes predictions to file using the best matrix object
    make_predictions_from_file("test.csv", minimum_mf)
    # Gets MAE and returns it to console
    mae = calc_error(minimum_mf)
    print()
    print(f"--- FINAL TEST MAE {mae} ---")


def get_mae(real_data, mf):
    """
    Get the Mean Absolute Error (MAE) between the real and generated predictions
    :param real_data: Real predicitions
    :param mf: MatrixFactoriser object that is used to get the predicted values
    :return: Mean Absolute Error
    """
    error = 0
    for u, i, real, timestamp in real_data:
        error += math.sqrt((real - mf.get_rating(u, i)) ** 2)
    return error / len(real_data)


def random_sample(list):
    """
    Randomly sample a List
    :param list: List to get random value from
    :return: Random value from List
    """
    return list[random.randint(0, len(list)-1)]


def create_mf(params):
    """
    Create a MatrixFactoriser object
    :param params: Parameters to pass to MatrixFactoriser (train_data, num_users, num_items, f, learning_rate,
                   regularisation_param, iterations)
    :return: MatrixFactoriser object
    """
    m = MatrixFactorisation()
    m.initialise_with_values(params)
    return m


def get_averages_and_best(maes, mfs):
    """
    Method to return the average values and best values for our error and matrix parameters
    :param maes:
    :param mfs:
    :return:
    """
    total_mae = 0
    best_mae = 0
    total_matrix_params = []
    best_matrix_params = []
    first_loop = True
    for mae, mf in zip(maes, mfs):
        if first_loop:
            best_mae = mae
            total_matrix_params = mf.get_matrix_params()
            first_loop = False
        else:
            total_matrix_params = add_params(total_matrix_params, mf.get_matrix_params())
            if mae < best_mae:
                best_mae = mae
                best_matrix_params = mf.get_matrix_params()
        total_mae += mae
    average_mae = total_mae/len(maes)
    average_mf = create_mf(divide_params(total_matrix_params, len(mfs)))
    best_mf = create_mf(best_matrix_params)

    return average_mae, best_mae, average_mf, best_mf


def generate_folds(data, k):
    """
    Method that generates k folds for cross validation
    :param data: Data to split into k folds
    :param k: Number of folds to split data into
    :return: Returns a list of lists of lists, with  each list element being a fold containing 2 sub lists,
             the first being a list of training data, and the second being a list of test data for that fold
             E.g., data = [1,2,3,4,5,6], k = 3 could return [[[1,2,3,4],[5,6]], [[3,4,5,6],[1,2]], [1,2,5,6],[3,4]]]
    """
    # Randomises data ordering so we get different folds each time
    np.random.shuffle(data)
    folds = []
    # Splits data into k chunks
    test_data_folds = np.array_split(data, k)
    outer_index = 0
    # Loops through the data list twice, setting the outer loop to being the test chunk, and all the other chunks
    # on the inner loop to be training data
    for test in test_data_folds:
        train = []
        for inner_index in range(len(test_data_folds)):
            if outer_index != inner_index:
                train += list(test_data_folds[inner_index])
        outer_index += 1
        folds.append([train, list(test)])
    return folds


def add_params(params1, params2):
    """
    Method to add sets of parameters together
    :param params1:
    :param params2:
    :return:
    """
    result = []
    for p1, p2 in zip(params1, params2):
        result.append(np.add(p1, p2))
    return result


def divide_params(params, divisor):
    """
    Method to divide a set of parameters by a divisor
    :param params: List of parameters to divide
    :param divisor: Divisor
    :return:
    """
    result = []
    for p1 in params:
        result.append(np.divide(p1, divisor))
    return result


def read_file(dir):
    """
    Method which reads all the training data from the file and returns this along with the maximum user and item
    :param dir: File path to read data from
    :return: Numpy Array, int, int : Training data, number of users, number of items
    """
    with open(dir, "r") as f:
        lines = csv.reader(f)
        data = []
        n_users = 0
        n_items = 0
        line_count = 0
        for row in lines:
            data.append([int(row[0])-1, int(row[1])-1, float(row[2]), int(row[3])])
            if int(row[0]) > n_users:
                n_users = int(row[0])
            if int(row[1]) > n_items:
                n_items = int(row[1])
            line_count += 1
    return np.array(data), n_users, n_items


def make_predictions_from_file(dir, mf):
    """
    Method used to read what predictions we had to make from a file, make the prediction, then write the result to
    results.csv
    :param dir: File path to read test data
    :param mf: MatrixFactoriser objects used to make prediction
    :return: None
    """
    predictions = []
    with open(dir, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            user_id = int(row[0])-1
            item_id = int(row[1])-1
            timestamp = int(row[2])
            prediction = round(mf.get_rating(user_id, item_id) * 2) / 2
            predictions.append([user_id+1, item_id+1, prediction, timestamp])
    with open("results.csv", "w", newline='') as w:
        writer = csv.writer(w)
        for p in predictions:
            writer.writerow(p)


def calc_error(mf):
    """
    Calculates the Mean Absolute Error (MAE) using the real values (stored in test_real.csv) compared to predicted values
    drawn from the ratings matrix
    :param matrix:
    :return: double : MAE
    """
    mae = 0
    row_count = 0
    with open("test_real.csv", "r") as f:
        read = csv.reader(f)
        for row in read:
            u = int(row[0])-1
            i = int(row[1])-1
            real = float(row[2])
            prediction = round(mf.get_rating(u, i) * 2) / 2
            mae += math.sqrt((real-prediction)**2)
            row_count += 1
    return mae/row_count


# Main method, the bit that actually ran
if __name__ == "__main__":
    random_search(1, 5, 1)
