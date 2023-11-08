import csv
import numpy as np

"""
This script converts the training data in its current format into a ratings matrix
The current form of the training data is rows of data, with each row containing one rating for one user on one item
This converts that to a matrix with axis of users and items, with each element therefore being a rating for a user
on a specific item
It reads the training data from 'train.csv', and saves the matrix to a file called 'matrix.csv'
"""

with open("train.csv", "r") as f:

    print("READING START")

    lines = list(csv.reader(f))
    max_user = 0
    max_item = 0
    for row in lines:
        if int(row[0]) > max_user:
            max_user = int(row[0])
        if int(row[1]) > max_item:
            max_item = int(row[1])
    print(max_item, max_user)
    matrix = np.zeros((max_user, max_item), dtype=np.float16)

    for row in lines:
        user = int(row[0]) - 1
        item = int(row[1]) - 1
        matrix[user][item] = np.float16(row[2])

    print("READING DONE")

    with open("matrix.csv", "w", newline='') as m:

        print("WRITING START")

        writer = csv.writer(m)
        matrix_size = len(matrix)
        print(matrix_size)
        row_count = 0
        for row in matrix:
            writer.writerow(row)
            row_count += 1
            if row_count % 1000 == 0:
                print("ROW {}".format(row_count))
                row_count = 0

        print("WRITING DONE")