"""
Designed for basic function, can replace the module data_processor step by step
"""

import os
import shutil
import time
from math import ceil

import numpy as np
from scipy.spatial import ConvexHull
from send2trash import send2trash


# file and folder functions
def folder_create(folderpath):
    """
    create the folder
    :param folderpath: the folderpath
    :return:
    """
    try:
        os.makedirs(folderpath)
    except:
        pass


def folder_create_with_curmonth(folderpath):
    """
    create the folder with current date (Year-Month) named
    :param folderpath: the folderpath input
    :return: folderpath
    """
    folderpath = folderpath + time.strftime("%Y_%b", time.localtime()) + '/'
    try:
        os.makedirs(folderpath)
    except:
        pass
    return folderpath


def folder_clean_recreate(folderpath):
    """
    clean the folder and recreate it
    :param folderpath: the folderpath
    :return:
    """
    try:
        shutil.rmtree(folderpath)
        os.makedirs(folderpath)
    except:
        os.makedirs(folderpath)


def folder_safeclean_recreate(folderpath):
    """
    clean the folder to trash and recreate it
    :param folderpath: the folderpath
    :return:
    """
    try:
        send2trash(folderpath)
        os.makedirs(folderpath)
    except:
        os.makedirs(folderpath)


# list processing functions
def list_nesting_remover(input_list, output_list=None):
    """
    to extract each element inside a list with deep nesting level
    :param input_list: (list/element) a list with multiple nesting level
    :param output_list: (list) a cumulated list during iteration
    :return: output_list: (list) a non-nesting list
    """
    # list_nesting_remover = lambda list_in: [list_out for i in list_in for list_out in list_nesting_remover(i)] if type(list_in) is list else [list_in]

    if output_list is None:
        output_list = []
    if type(input_list) is list:
        for i in input_list:
            output_list = list_nesting_remover(i, output_list)
    else:
        output_list.append(input_list)
    return output_list


def random_split(data, split_ratio):
    """
    lighter version of torch.torch.utils.data.random_split
    split dataset into 2 subsets at 1st dimension
    :param data: (1D-list, ndarray) the dataset need to be split
    :param split_ratio: (float) the ratio for split
    :return: subdata1: data subset1
             subdata2: data subset2
    """
    # convert input data to numpy array
    data = np.array(data)
    # get split index of the data
    subdata1_idx = np.random.choice(len(data), int(split_ratio * len(data)), replace=False)
    subdata2_idx = np.delete(np.arange(len(data)), subdata1_idx)
    subdata1 = data[subdata1_idx]
    subdata2 = data[subdata2_idx]
    return subdata1, subdata2


def dataset_split(data, split_ratio, random=True):
    """
    lighter version of torch.torch.utils.data.random_split
    split dataset into multiple subsets at 1st dimension randomly by default
    :param data: (1D-list, ndarray) the dataset need to be split
    :param split_ratio: (tuple/list - float) the ratio for split, e.g., (0.7, 0.2, 0.1)
    :param random: (Boolean) the enable for random the dataset
    :return: subdata_list: data subsets
    """
    # convert input data to numpy array
    data = np.array(data)
    datalen = len(data)
    subdata_list = []

    split_ratio_cum = np.cumsum(split_ratio)
    if split_ratio_cum[-1] < 1:
        split_ratio_cum = np.concatenate([split_ratio_cum, [1]])

    if random:
        np.random.shuffle(data)

    ra1 = 0
    for ra2 in split_ratio_cum:
        # get split index of the data
        subdata_idx = np.arange(round(datalen * ra1), round(datalen * ra2), 1)
        subdata = data[subdata_idx]

        # update split index
        ra1 = ra2

        # append the subdata to the list
        subdata_list.append(subdata)

    # remove empty nparray
    if len(subdata_list[-1]) == 0:
        subdata_list.pop(-1)

    return subdata_list


# numpy 2D data processing functions
def np_get_idx_bool(data, idx, range_lim, mode=1):
    """
    only one idx can be processed in one call
    :param data: (ndarray) data_numbers(n) * channels(c)
    :param idx: (int) the idx number
    :param range_lim: (tuple/int/float) (bottom_lim, upper_lim) element can be None, the range for preserved data
    :param mode: (int) 0-[min, max], 1-[min, max), 2-(min, max], 3-(min, max), include boundary or not
    :return: preserved_index: (ndarray-bool) data_numbers(n)
             removed_index: (ndarray-bool) data_numbers(n)
    """
    # initialize the index array
    preserved_index = np.ones(len(data), dtype=bool)
    removed_index = np.zeros(len(data), dtype=bool)

    if range_lim is not None:
        if type(range_lim) is tuple or type(range_lim) is list:  # expect list type
            if range_lim[0] is not None:
                if mode == 0 or mode == 1:
                    index = data[:, idx] >= range_lim[0]
                else:
                    index = data[:, idx] > range_lim[0]
                # update the index
                preserved_index = preserved_index & index
                removed_index = removed_index | ~index
            if range_lim[1] is not None:
                if mode == 0 or mode == 2:
                    index = data[:, idx] <= range_lim[1]
                else:
                    index = data[:, idx] < range_lim[1]
                # update the index
                preserved_index = preserved_index & index
                removed_index = removed_index | ~index
        else:  # expect int/float type
            index = data[:, idx] == range_lim
            # update the index
            preserved_index = preserved_index & index
            removed_index = removed_index | ~index
    return preserved_index, removed_index


def np_filter(data, idx, range_lim, mode=1):
    """
    only one idx can be processed in one call
    :param data: (ndarray) data_numbers(n) * channels(c)
    :param idx: (int) the idx number
    :param range_lim: (tuple/int/float) (bottom_lim, upper_lim) element can be None, the range for preserved data
    :param mode: (int) 0-[min, max], 1-[min, max), 2-(min, max], 3-(min, max), include boundary or not
    :return: data_preserved: (ndarray) data_numbers(n) * channels(c)
             data_removed: (ndarray) data_numbers(n) * channels(c)
    """
    preserved_index, removed_index = np_get_idx_bool(data, idx=idx, range_lim=range_lim, mode=mode)
    # get data and noise
    data_preserved = data[preserved_index]
    data_removed = data[removed_index]
    return data_preserved, data_removed


def SNRV_filter(data_points, SNRV_threshold):
    """
    :param data_points: (ndarray) data_numbers(n) * channels(c=5)
    :param SNRV_threshold: (dict) the SNR threshold
    :return: data_points: (ndarray) data_numbers(n) * channels(c=5)
             noise: (ndarray) data_numbers(n) * channels(c=5)
    """
    # remove points with low energy strength
    data_points, noise = np_filter(data_points, idx=4, range_lim=SNRV_threshold['range'])

    # identify the noise with speed
    if len(noise) > 0 and SNRV_threshold['speed_none_0_exception']:
        noise, noise_with_speed = np_filter(noise, idx=3, range_lim=0)
        data_points = np.concatenate([data_points, noise_with_speed])
    return data_points, noise


def np_2D_set_operations(dataA, dataB, ops='intersection'):
    """
    set operations for 2D ndarray, provide intersection, subtract, union
    :param dataA: (ndarray) data_numbers(n) * channels(c)
    :param dataB: (ndarray) data_numbers(m) * channels(c)
    :param ops: (str) operation name
    :return: data_preserved: (ndarray) data_numbers(n) * channels(c)
    """
    maskA = np.all(dataA[:, np.newaxis] == dataB, axis=-1).any(axis=1)
    maskB = np.all(dataB[:, np.newaxis] == dataA, axis=-1).any(axis=1)

    intersection = dataA[maskA]  # or dataB[maskB]
    dataA_subtract = dataA[~maskA]
    dataB_subtract = dataB[~maskB]

    if ops == 'intersection':
        return intersection
    elif ops == 'subtract':
        return dataA_subtract
    elif ops == 'subtract_both':
        return dataA_subtract, dataB_subtract
    elif ops == 'exclusive_or':
        exclusive_or = np.concatenate([dataA_subtract, dataB_subtract])
        return exclusive_or
    elif ops == 'union':
        union = np.concatenate([dataA_subtract, intersection, dataB_subtract])
        return union
    else:
        raise ValueError('ops can be intersection, subtract, subtract_both, exclusive_or, union')


def np_2D_repeated_points_removal(data, idxes=None):
    """
    remove the repeated points, by default compare all indexes (only remove if values in all indexes are repeated),
    compare the designated indexes if indexes are listed (only remove if values in listed indexes are repeated)
    :param data: (ndarray) data_numbers(n) * channels(c)
    :param idxes: (tuple/list) indexes used to compare the data, if None then means all indexes will be used to compare
    :return: points_unique (ndarray) data_numbers(n) * channels(c)
    """
    # lower the precision to speed up
    data = data.astype(np.float16)
    # data = np.around(data, decimals=2)  # not working when the number is too big more than 500

    # remove repeated points
    if idxes is None:
        points_unique = np.unique(data, axis=0)
    elif type(idxes) is tuple or type(idxes) is list:
        points_unique = np.empty([0, data.shape[1]])
        # get all rows according to the axes and do unique
        data_sub = data[:, idxes]
        data_sub = np.unique(data_sub, axis=0)
        # extract from the original data
        for ds in data_sub:
            temp = data
            # for each unique data_sub, relocate it in data
            for i, idx in enumerate(idxes):
                temp, _ = np_filter(temp, idx=idx, range_lim=ds[i])
            # get average values for the rest axis
            points_unique = np.concatenate([points_unique, np.average(temp, axis=0)[np.newaxis]])
    else:
        raise Exception('axes type is not supported')
    return points_unique


def np_window_sliding(data, window_length, step):
    """
    window slide and stack at 1st dimension of data,
    if the last window step can not be formed due to insufficient rest data, it will be dropped.
    :param data: (ndarray) data_numbers(n) * channels(c)
    :param window_length: (int) the stacked length for 2nd dimension time
    :param step: (int) >0, the number of data skipped for each sliding
    :return: data_stacked: (ndarray) data_numbers(n) * time(t) * channels(c)
    """
    total_step = ceil((len(data) - window_length) / step)
    if total_step > 0:
        data_stacked = np.ndarray([0, window_length] + list(data.shape)[1:])
        for i in range(total_step):
            data_window = data[i * step:i * step + window_length][np.newaxis]
            data_stacked = np.concatenate([data_stacked, data_window])
    else:
        raise Exception(f'The data length is not long enough!')
    return data_stacked


def np_2D_row_removal(database, data_remove):
    """
    remove those data rows in the data_remove from the database
    :param database: (ndarray) data_numbers(n) * channels(c), original database
    :param data_remove: (ndarray) data_numbers(n) * channels(c), need to be removed from database
    :return: database: (ndarray) data_numbers(n) * channels(c), updated database
    """
    # locate the data_remove in database first and get boolean index
    database_index = np.zeros(len(database), dtype=bool)
    for d in data_remove:
        # locate one data row which needs to be removed in database
        row_index = np.ones(len(database), dtype=bool)
        for i in range(data_remove.shape[1]):
            idx, _ = np_get_idx_bool(database, idx=i, range_lim=d[i])
            row_index = row_index & idx
        database_index = database_index | row_index
    # reverse the index to remove data_remove and output database
    return database[~database_index]


def boundary_calculator(points, idx):
    """
    multiple index can be processed in one call
    :param points: (ndarray) data_numbers(n>0) * channels(c)
    :param idx: (int/tuple/list/range) the index number
    :return: result: (tuple-tuple) boundaries, (min, max)
    """
    result = []
    idx_list = [idx] if type(idx) == int else idx
    for idx in idx_list:
        dmin, dmax = points[:, idx].min(), points[:, idx].max()
        result.append((dmin, dmax))
    return tuple(result) if len(result) > 1 else result[0]


def convexhull(cluster):
    """
    :param cluster: (ndarray) data_numbers(n) * channels(c>=3) for cluster data
    :return: (ndarray) data_numbers(16) * channels(3) for hull vertices
    """
    try:  # just in case of insufficient points or all points in single line
        vertices_index = ConvexHull(cluster[:, 0:3]).vertices
        vertices_index = np.concatenate([vertices_index, vertices_index[0:1]])  # connect end to end for drawing closed shape
    except:
        vertices_index = []
    return cluster[vertices_index]

def cubehull(cluster, *args):
    """
    :param cluster: (ndarray) data_numbers(n) * channels(c>=3) for cluster data
    :return: (ndarray) data_numbers(16) * channels(3) for hull vertices
    """
    try:
        xboundary, yboundary, zboundary = args
    except:
        xboundary, yboundary, zboundary = boundary_calculator(cluster, idx=range(3))
    xmin, xmax = xboundary
    ymin, ymax = yboundary
    zmin, zmax = zboundary
    x = [xmin, xmax, xmax, xmin, xmin, xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmax, xmax, xmin, xmin]
    y = [ymin, ymin, ymax, ymax, ymin, ymin, ymin, ymax, ymax, ymin, ymin, ymin, ymax, ymax, ymax, ymax]
    z = [zmin, zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax, zmax, zmax, zmin, zmin, zmax, zmax, zmin]
    return np.array([x, y, z]).T


if __name__ == '__main__':
    # dataset = [2, 6, 9, 3, 7, 8, 10, 199, 10]
    dataset = np.arange(0, 30, 1).reshape(10, -1)
    a, b, c = dataset_split(dataset, (0.7, 0.2, 0.1), True)
    pass
