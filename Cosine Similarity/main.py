import logging
import csv

import math
import numpy as np
from numpy.linalg import norm

np.seterr(all="ignore")
# Setting up logger
LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('logging started')
# Dictionary to map user + rating to timestamp
timestamp_dict = {}

def calculate_user_averages(train_data, n_users):
    """
    Method to find each user's average ratings
    Have arrays where each index represents a user
    Essentially sum each user's ratings and put them in sum_of_user_ratings
    Count how many ratings each user does and put that in number_of_ratings_per_user
    Then, do elementwise division of sum_of_user_ratings by number_of_ratings_per_user to get averages
    Store this in average_ratings
    :param train_data: rows of training data
    :param n_users: number of users
    :return: List : Elements are the average of each user's rating, and the user number is  the index.
    """
    logger.info("Calculating Averages : START")
    # Arrays to find the average user ratings
    sum_of_user_ratings = np.zeros(n_users, dtype=np.float16)
    number_of_ratings_per_user = np.zeros(n_users, dtype=np.float16)

    for row in train_data:
        user_id = int(row[0])
        rating = float(row[2])
        sum_of_user_ratings[user_id - 1] += rating
        number_of_ratings_per_user[user_id - 1] += 1.0

    # Calculates the average ratings by doing elementwise division of the sum's of the ratings
    # per user by the number of ratings made by each user
    average_ratings = np.divide(sum_of_user_ratings, number_of_ratings_per_user)

    logger.info("Calculating Averages : DONE")
    return average_ratings

def format_training_data(train_data, n_users, n_items, averages):
    """
    Method to format data into preferred layout (item centered)
    Will be arranged in vectors where each vector is representing an item
    The entries in the vector are a rating (bias removed), and the index they are at correlates to the user that gave
    that rating for the item which the vector is about
    E.g. [[3,5,3],[1,5,3],[2,3,4]] , the first horizontal vector [3,5,3] is for item ID = 1
    and the first entry (3) is the rating user ID = 1 gave for that item
    Doing this so can perform dot product later instead of manual calculation using for loops
    :param train_data: List of rows of user data (row = [user, item, prediction, timestamp])
    :param n_users: number of users
    :param n_items: number of items
    :param averages: List of average ratings for each user
    :return: List, List : List of item data, list of user data
    """
    logger.info("Formatting Training Data : START")
    # Sets up 2D array for new format of data
    formatted_data_for_items = np.zeros((n_items, n_users), dtype=np.float16)
    formatted_data_for_users = np.zeros((n_users, n_items), dtype=np.float16)

    # Loops through each row and adds the adjusted ratings to the formatted_data array
    # in the format described at top of method
    for row in train_data:
        user_id = int(row[0])
        item_id = int(row[1])
        rating = float(row[2])
        user_average = averages[user_id - 1]
        formatted_data_for_items[item_id - 1][user_id - 1] = rating - user_average
        formatted_data_for_users[user_id - 1][item_id - 1] = rating

    logger.info("Formatting Training Data : DONE")
    return formatted_data_for_items, formatted_data_for_users


def train_model(train_data_formatted, n_users, n_items, user_averages):
    """
    Training function for cosine similarity recommender system algorithm
    :param train_data_formatted:
    :param n_users:
    :param n_items:
    :param user_averages:
    :return: List : item similarity matrix
    """
    logger.info("Training : START")
    # initialize vector and matrix ready for population
    item_sim_matrix = np.full((n_items, n_items), fill_value=1.0, dtype=np.float16)

    # We loop through each item vector, and compare it to every other item vector we have
    # Note, we are efficient here as we only calculate the similarity one-way, as it is symmetric.
    # E.g., if we calculate sim(i1,i2), we know it is the same as sim(i2,i1), so we just put it in both spots
    # of the similarity matrix rather than calculate it again.
    for i1_index in range(len(train_data_formatted)):
        for i2_index in range(i1_index + 1, len(train_data_formatted)):
            # Each vector, i1 and i2, represents all ratings for that item, with the index
            # of each rating correlating to a user
            # E.g., i1 = [1,4,5] would mean user1 rated i1 as 1, user2 rated it 4 and user3 gave it a 5
            #
            # NOTE: these ratings have already been adjusted for the user's average rating
            i1 = train_data_formatted[i1_index]
            i2 = train_data_formatted[i2_index]

            # For the adjusted cosine similarity between i1 and i2, we have to multiply each user's adjusted rating for
            # i1 and i2, and sum this over all users who have rated both items
            # This is captured by the dot product of our item vectors
            # E.g., i1 = [2,4], i2 = [4,5]. If we multiply u1 ratings for i1 and i2 together we get 2*4 = 8, then
            # we do the same for u2, 4*5 = 20, then sum these giving 28. This is the same as performing
            # the dot product: [2,4] . [4,5] = 2*4 + 4*5 = 28
            top_sum = np.dot(i1, i2)

            # This next section is not part of the equation necessarily but is needed because of how I am storing
            # the data
            # Basically, the bottom part of the cosine similarity equation requires us to find the norm of
            # each item vector, but only for ratings where both users have rated both i1 and i2.
            # This means if there is a 0 in one position in one of the vectors, say i1, we do not want to include
            # the rating in the same index in the other vector, say i2. By include here I mean include in the norm
            # calculation
            # E.g., i1 = [0,2,0], i2 = [1,4,5]. Since u1 and u3 have not rated i1 (0 at indexes 0 and 2 in i1),
            # we do not want to include the ratings of u1 and u3 for i2 in the norm calculations.
            #
            # To achieve this, I have had to multiply each item vector together element-wise, then divide through by
            # each vector again to get each vector with zeros in positions where both users did not rate both items.
            # E.g., for the values above, i1 x i2 = [0,8,0]. To get i1, we do [0,8,0]/i2 = [0,2,0]. And for i2 =
            # [0,8,0]/i1 = [0,4,0]
            #
            # NOTE: the np.nan_to_num method just swaps out the NaN values to 0s, as the element-wise division
            # 0/0 gives the result NaN, but we want this to be 0. Also, we check if the element-wise multiplication
            # results in a zero vector, because if it does, we know there is 0 similarity so can skip the other
            # calculations

            multiplied = np.multiply(i1, i2)
            if np.sum(multiplied) != 0:
                adjusted_i1 = np.nan_to_num(np.divide(multiplied, i2))
                adjusted_i2 = np.nan_to_num(np.divide(multiplied, i1))

                # Finally we perform the cosine similarity equation, taking the top_sum we calculated earlier
                # and dividing it by the norms of both i1 and i2 vectors, where they have been adjusted so that they
                # only contain values where both users have rated both items
                similarity_i1_i2 = top_sum / (norm(adjusted_i1) * norm(adjusted_i2))

            else:
                similarity_i1_i2 = 0

            # After calculating the similarity of i1 and i2, we enter this value into the similarity matrix
            # We enter into 2 positions as this matrix is symmetric.
            item_sim_matrix[i1_index][i2_index] = similarity_i1_i2
            item_sim_matrix[i2_index][i1_index] = similarity_i1_i2

    logger.info("Training : DONE")
    # Return the populated item to item similarity matrix
    return item_sim_matrix


def predict_ratings(test_data, item_sim_matrix, user_rating_data, user_averages):
    """
    Infer function for cosine similarity recommender system algorithm
    :param test_data: Test data retrieved from csv file, each row is a row in csv file
    :param item_sim_matrix: item similarity matrix. where each row is and item's similarity to all other items
    :param user_rating_data: matrix where each row is a user's ratings for all items.
    :param user_averages: Vector containing average rating for each user
    :return: List, rows of predictions
    """
    logger.info("Predicting : START")
    predictions = []

    # Making a prediction for each row in the test data
    for row in test_data:
        # Gathers all information needed from the row; user to make prediction for, item we are predicting,
        # and timestamp of prediction
        user = int(row[0])  # u
        new_item = int(row[1])  # i
        timestamp = int(row[2])
        # Gets the average rating for the user we are making a prediction for
        user_average = user_averages[user - 1]
        # Gets the similarity against all other items of the item we are making a prediction for
        # Similarity vector
        item_similarity_vector = item_sim_matrix[new_item - 1]
        # Gets rating vector for the user u
        user_rating_vector = user_rating_data[user - 1]

        # This section is to subtract the user's average away from their ratings vector
        # Have the subtractor vectorised function so that any ratings not made by the user (rating = 0) are not included
        # If this was not included, any non-ratings would be included in the prediction
        subtractor = np.vectorize(lambda x, y: x - y if x != 0 else 0)
        average_user_rating_vector = subtractor(user_rating_vector, user_average)

        try:
            # This is where the prediction calculation is made, for user u and item i
            # The dot product of the similarity vector with the user's rating vector is
            # the top the equation, meaning it's the sum over similarity between item i and
            # another item i' multiplied by the user's rating for that item i'. This scales each rating of the user
            # by the similarity between the items, meaning more similar items will scale the rating more so will get
            # a higher prediction. The dividing over the norm of the item_similarity_vector normalises the result.
            predction = (np.dot(item_similarity_vector, average_user_rating_vector) / (norm(item_similarity_vector)))
            # Currently, prediction is a large float value and want and small float rounded to closest 0.5
            # This is done by first adding back the user average to the prediction, then multiplying it by 2, then
            # rounding it to the nearest whole number, then dividing that result by 2.
            final_prediction = round((predction + user_average) * 2) / 2
        except (ValueError, OverflowError):
            final_prediction = 0
        # Prediction added to the list of predictions
        predictions.append([user, new_item, final_prediction, timestamp])

    logger.info("Predicting : DONE")
    # return the predicted ratings matrix
    return predictions


def serialize_predictions(output_file, prediction_matrix):
    """
    Serialize a set of predictions to file
    :param output_file: Filename to output results to
    :param prediction_matrix: Matrix containing predictions
    :return: None
    """
    logger.info("Writing : START")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in prediction_matrix:
            writer.writerow(row)
    logger.info("Writing : DONE")


def load_data(file):
    """
    Load a set of ratings from file
    :param file: filename containing data to load
    :return: List, int, int : List containing rows of data from file, number of users, number of items
    """

    #
    # Currently in the format of:
    # [[user_id, item_id, user_rating, timestamp]]
    #
    # Code to read the raw data from the csv file
    # Reads the file in csv format and saves each row as a list
    # Each list row is added to the raw_data list of lists
    # Also fills the timestamp_dict with each row
    # This is to correlate each timestamp to each row, as I am going to reformat the data, but need to be able
    # to correlate back to the timestamp at the end for submission
    logger.info("Loading : START")

    raw_data = []

    n_users = 0
    n_items = 0
    with open(file) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for row in reader:
            user_id = int(row[0])
            item_id = int(row[1])
            # This assumes there is are no ID gaps
            # I.e., IDs are assigned by starting at 1 and incrementing by one per new item/user
            n_users = user_id if user_id > n_users else n_users
            n_items = item_id if item_id > n_items else n_items
            raw_data.append(row)
            timestamp_dict[int(row[len(row) - 1])] = row

    logger.info("Loading : DONE")
    return raw_data, n_users, n_items


def evaluate(results_file, true_results):
    """
    Method to calculate Mean Absolute Error (MSE) and Root Mean Squared Error (RMSE)
    :param results_file: Filename containing the predicted ratings from the program
    :param true_results: Filename containing true results
    :return: None
    """
    MAE = 0
    RMSE = 0
    with open(results_file) as results:
        with open(true_results) as true:
            reader1 = csv.reader(results)
            reader2 = csv.reader(true)
            counter = 0
            for row1, row2 in zip(reader1, reader2):
                MAE += math.sqrt((float(row1[2]) - float(row2[2])) ** 2)
                RMSE += (float(row1[2]) - float(row2[2])) ** 2
                counter += 1

    MAE = MAE / counter
    RMSE = math.sqrt(RMSE / counter)

    logger.info(f"MAE : {MAE}")
    logger.info(f"RMSE : {RMSE}")


if __name__ == '__main__':
    logger.info('COSINE SIMILARITY RECOMMENDER SYSTEM')

    # load test and training data into memory
    train_data, n_users, n_items = load_data(file='train.csv')
    user_averages = calculate_user_averages(train_data, n_users)
    formatted_data_for_items, formatted_data_for_users = format_training_data(train_data, n_users, n_items,
                                                                              user_averages)

    # Call the train function to learn similarity weights for cosine similarity recommender system algorithm
    item_sim_matrix = train_model(train_data_formatted=formatted_data_for_items, n_users=n_users,
                                  n_items=n_items, user_averages=user_averages)

    # Call the infer function to execute the cosine similarity recommender system algorithm
    test, unused_1, unused_2 = load_data(file='test.csv')
    pred = predict_ratings(test, item_sim_matrix, formatted_data_for_users, user_averages)

    # Serialize the rating predictions to file
    serialize_predictions(output_file='test_results.csv', prediction_matrix=pred)
    logger.info('Predictions saved to file submission.csv')

    # Evaluates the predictions made into test_results.csv against the correct value stored in test_real.csv
    # Logs the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to console
    evaluate("test_results.csv", "test_real.csv")



