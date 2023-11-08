import numpy as np
import math


class MatrixFactorisation:
    """
    Class to perform Matrix Factorisation
    """

    def __init__(self, train_data=None, num_users=0, num_items=0, f=0, learning_rate=0, regularisation_param=0, iterations=0):
        """
        Constructor which initialises class parameters

        :param train_data : List of training data [user, item, rating, timestamp]
        :param num_users : Number of users
        :param num_items : Number of items
        :param f : Number of latent dimensions
        :param learning_rate : Learning rate for stochastic gradient descent
        :param regularisation_param : Regularisation parameter for regularisation
        :param iterations : Number of iterations for training
        :return: None
        """

        self.num_users = num_users
        self.num_items = num_items
        self.train_data = np.array(train_data)
        self.num_user = num_users
        self.num_items = num_items
        self.f = f
        self.learn_rate = learning_rate
        self.reg_param = regularisation_param
        self.iterations = iterations
        self.p, self.q, self.b_u, self.b_i, self.b = None, None, None, None, None


    # Method to initialise the latent feature matrices and biases so that they can be updated in SGD
    def initialise(self):
        """
        Method to initialise the starting values for the latent factor matrices (p and q), as well as the
        corresponding weights and biases for these matrices (b_u, b_i) and an overall bias (b). The matrices
        are initialised using the normal distribution, the specific biases are initialised as zero, and the
        global bias is initialised as the mean of the training data

        :return: None
        """

        # Initialize user and item latent feature matrices
        self.p = np.random.normal(scale=1. / self.f, size=(self.num_users, self.f))
        self.q = np.random.normal(scale=1. / self.f, size=(self.num_items, self.f))
        # Initialise specific biases for each user and item feature matrix, using standard normal distribution
        # There will be one bias to each feature vector within each of the user and item feature matrices
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        # Initialise an overall bias to keep ratings centralised around the average
        self.b = np.mean(self.train_data, axis = 0)[2]


    def initialise_with_values(self, params):
        """
        Method to initialise the whole class with parameters, so that we can use an object to get ratings of our best
        found parameters (used for k-fold-cross validation and random search)

        :param params: List of values to set the parameters to ([p, q, b_u, b_i, b])
        :return: None
        """

        self.p, self.q, self.b_u, self.b_i, self.b = params[0], params[1], params[2], params[3], params[4]

    def train(self):
        """
        This method runs the stochastic gradient descent loop in order to form the latent feature matrices

        :return: None
        """

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.train_data)      # Shuffle data
            self.sgd()                              # Perform SGD step
            mae = self.mae()                        # Calculate MAE
            training_process.append((i, mae))       # Save MAE

        return training_process[-1][1]              # Return MAE for training process

    def mae(self):
        """
        Calculates the Mean Absolute Error (MAE) between predicted value and real value in training data

        :return: double : MAE
        """

        error = 0
        for u, i, real, timestamp in self.train_data:
            error += math.sqrt((real - self.get_rating(u, i))**2)
        return error / len(self.train_data)

    def sgd(self):
        """
        Method to perform 1 step of stochastic gradient descent

        :return: None
        """

        # Loop through each item in the training data
        for user, item, real, timestamp in self.train_data:
            # Having to round the u and i as I kept getting errors where sometimes the u and i were floats
            u = int(round(user))
            i = int(round(item))
            # Compute prediction and error of that prediciton
            prediction = self.get_rating(u, i)
            e = (real - prediction)

            # Update user and item feature biases
            # Each bias, specific to a user or item, is updated using the learning rate parameter (used to control
            # the rate of learning of every update) multiplied by the gradient, which is given by the error multiplied
            # by the opposite bias, subtracted by the original value for this bias mutlipled by the regularisation
            # parameter, which controls overfitting
            self.b_u[u] += self.learn_rate * ((e * self.b_i[i]) - self.reg_param * self.b_u[u])
            self.b_i[i] += self.learn_rate * ((e * self.b_u[u]) - self.reg_param * self.b_i[i])

            # Update user and item latent feature matrices
            # These are done in the same way as above but with the feature matrices, self.p and self.q,
            # instead of the bias vectors, self.b_u and self.b_i
            self.p[u] = np.add(self.p[u],
                               np.multiply(self.learn_rate,
                                           np.subtract(np.multiply(e, self.q[i, :]),
                                                       np.multiply(self.reg_param, self.p[u, :]))))
            self.q[i] = np.add(self.q[i],
                               np.multiply(self.learn_rate,
                                           np.subtract(np.multiply(e, self.p[u, :]),
                                                       np.multiply(self.reg_param, self.q[i, :]))))

    def get_rating(self, u, i):
        """
        Method to get a predicted rating

        :param u: User Index to get rating for
        :param i: Item Index to get rating for
        :return: double : predicted rating
        """

        u = int(u)
        i = int(i)
        prediction = self.b + self.b_u[u] + self.b_i[i] + (self.p[u].dot(self.q[i].T))
        return prediction

    def get_matrix_params(self):
        """
        Get the parameters for the Matrix Factoriser object.
        Used in the random search and cross validation algorithms

        :return: List : [p, q, b_u, b_i, b]
        """

        return [self.p, self.q, self.b_u, self.b_i, self.b]


