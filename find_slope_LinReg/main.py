from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

def find_best_slope_and_intercept(xs, ys):
    # m = (x * y - xy)/((x)^2 - (x^2))
    # every character in the formula represents the average of the dataset
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs) * mean(xs)) - mean(xs*xs ))) # doesn't support:  mean(xs)^2 or mean(xs)**2

    # b = y - mx; y&x are the mean of the array
    b = mean(ys) - m * mean(xs)

    return m,b

def squared_error(ys_org, ys_line):
    # y_org is the real value that get from the data set
    # y_line is the value you get from the line of the linear regression

    # we return the squared gap of the error
    return sum((ys_line - ys_org)**2)

def coefficient_of_determinition(ys_org, ys_line):
    y_mean_line = [mean(ys_org) for y in ys_org] #for every y in the original line
    squared_error_regr = squared_error(ys_org, ys_line)
    squared_error_y_mean = squared_error(ys_org, y_mean_line)

    # fromula: r^2 = 1- (SE(y_regresion)) / (SE(y_mean))
    return 1 - (squared_error_regr / squared_error_y_mean)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    style.use('fivethirtyeight')
    # example data
    xs = np.array([1,2,3,4,5,6], dtype=np.float64)
    ys = np.array([5,4,6,5,6,7], dtype=np.float64)
    m,b = find_best_slope_and_intercept(xs, ys)

    # create the vector Y depends on every x value
    regression_line = [m*x + b for x in xs]

    predict_x = 8
    predict_y = (m * predict_x + b) # value that doesn't exist and we want to predict

    # r^2 who help us to determine if the values better or not
    # we want r^2 to be bigger as can because it represnt the error value is small
    r_squared = coefficient_of_determinition(ys, regression_line)

    plt.scatter(xs,ys) # place dot of the real value
    plt.plot(xs, regression_line) # draw the linear regression line
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
