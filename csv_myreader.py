import csv
import numpy as np

def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_csv_data(size=None):

    """
    :return: X = [(height, weight), (height, weight), ... ], y = [sex, sex, ... ]
    """

    dt = np.dtype('<U32')
    with open('data.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rows = list(rows)
        rows = list(map(lambda r: r[0].split(','), rows))

        print("Number of data: ", len(rows))
        print(*rows[:5], sep='\n')
        print("...")

    X = []
    y = []

    middle = [[0, 0], [0, 0]]
    for row in rows[1:size]:
        a, b = float(row[-2]) / 20, float(row[-1]) / 50
        c = int(row[3]) - 1
        middle[c][0] += a
        middle[c][1] += b

    middle[0][0] /= size
    middle[0][1] /= size
    middle[1][0] /= size
    middle[1][1] /= size

    for row in rows[1:size]:
        a, b = float(row[-2]) / 20, float(row[-1]) / 50
        c = int(row[3]) - 1

        if dist(middle[c], [a, b]) > 9: continue

        X.append([a, b])
        y.append(c)

    print(len(X), len(y))
    return (np.array(X), np.array(y))

d = get_csv_data(4000)
print(d[0][:5], d[1][:5])
print(len([i for i in d[1] if i == 1]))