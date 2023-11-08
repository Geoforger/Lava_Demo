import numpy as np
import re
import pandas as pd

def linear_accel(start, dt, rate):
    return start + (rate * dt)

def distance_from_accel(v, t, a):
    return v*t + 1/2*a*t^2


def time_from_accel(start_velocity, acceleration, distance):
    """
    Function to calculate total time from starting velocity, constant acceleration and distance traveled using quadratic equation

    Parameters
    ----------

    v (int): Starting velocity in mm/s
    a (int): Constant acceleration in mm/s/s
    d (int): Distance traveled in mm

    Returns
    -------

    root1 or root2 (float): Length of the sample in s
    """
    # Calculate the coefficients for the quadratic equation
    a = 0.5 * acceleration
    b = start_velocity
    c = -distance

    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Check if the discriminant is non-negative (real solutions)
    if discriminant >= 0:
        # Calculate the two possible solutions for time
        root1 = (-b + np.sqrt(discriminant)) / (2*a)
        root2 = (-b - np.sqrt(discriminant)) / (2*a)

        # Choose the positive root (time cannot be negative)
        if root1 >= 0:
            return root1
        elif root2 >= 0:
            return root2
        else:
            # If both roots are negative, no valid solution
            return None
    else:
        # If the discriminant is negative, no real solutions
        return None
    
def calculate_pooling_dim(input_dim, kernel_dim, stride, order):
    """
    Private method to calculate the size of a dimension after pooling

    Arguments
    ----------
    input_dim (int): Size of input dimension
    kernel_dim (int): Size of the pooling kernel along the same dimension
    stride (int): Stride of the pooling operation
    order (int): Number of pooling operations being performed
    """
    return int(((input_dim - kernel_dim) / stride) + (1 * order)) + 1

def floats_from_string(string):
    """
    Function to return a list of float values from a string
    
    Arguments
    ----------
    string (str): String to extract floats from

    Returns
    ---------
    f (list): List of floating point values from the string
    """
    l = re.findall("\d+\.\d+", string)
    f = [float(fl) for fl in l]
    
    return f

def nums_from_string(string):
    """
    Function to return a list of float values from a string
    
    Arguments
    ----------
    string (str): String to extract floats from

    Returns
    ---------
    i (list): List of int values from the string
    """
    l = re.findall(r'\d+', string)
    i = [int(a) for a in l]
    
    return i

def log_data(path, data):
    """
    Function to save the label, prediction and confidence of system in a csv file
    
    Arguments
    ----------
    path (str): Path to csv save location
    data (dict): Dict of data to append to csv
    """
    data_frame = pd.DataFrame(data=data)

    # Save csv file
    with open(path, 'a') as f:
        data_frame.to_csv(f, index=False, header=True)
