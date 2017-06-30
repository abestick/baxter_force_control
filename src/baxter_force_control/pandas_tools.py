import numpy as np
import pandas as pd
from kinmodel import GeometricPrimitive
import rospy


def dot_wrapper(args):
    """A wrapper function for performing the dot product of all columns for each row of a DataFrame"""
    args = [arg / np.linalg.norm(arg) for arg in args]
    return np.abs(np.dot(*args))


def hstack_wrapper(args):
    """A wrapper for horizontally concatenating array-like columns for each row of a DataFrame"""
    return pd.Series(np.hstack(args))


def one_d_columns(df, columns=None):
    """
    Takes a data frame with some array-like columns and separates it into multiple columns with only scalars
    :param pd.DataFrame df: A data frame whose columns have elements with higher dimensional objects
    :param columns: optional list of column names
    :return: a new DataFrame with each column containing a single value
    """
    new_df = df.apply(hstack_wrapper, axis=1)

    if columns is None:
        columns = []
        for name, obj in df.loc[len(new_df)-1].iteritems():
            if isinstance(obj, GeometricPrimitive):
                columns.extend(obj.names('%s_'%name))

            elif isinstance(obj, np.ndarray):
                columns.extend(range(len(obj)))

            else:
                columns.append(name)

    new_df.columns = columns

    return new_df


def dot_product(df, first, second):
    """
    Performs the row-wise dot product between two array-like columns of a DataFrame
    :param pd.DataFrame df:the data frame that contains the columns
    :param first: the first column name
    :param second: the second column name
    :return: A new DataFrame containing only the dot product of the two columns
    """
    return df[[first, second]].apply(dot_wrapper, axis=1)


def dot_products(df, pairs, add=False):
    """
    Performs the row-wise dot product for multiple pairs of array-like columns of a DataFrame
    :param pd.DataFrame df:the data frame that contains the columns
    :param pairs: an iterable of pairs of column names
    :param bool add: if set to true, the returned data frame includes the original
    :return:
    """
    if add:
        new_df = df.copy()
    else:
        new_df = pd.DataFrame()

    for first, second in pairs:
        new_df['<%s,%s>'%(first, second)] = dot_product(df, first, second)
    return new_df
