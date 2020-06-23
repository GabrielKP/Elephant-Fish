import numpy as np
import pandas as pd
import math
from functions import getAngle, getDistance, readClusters
from itertools import chain
from reader import *
from sklearn.cluster import KMeans

def getLocomotion(np_array, path_to_save_to, saveToFile=True, mode="radians"):
    """
    This function expects to be given a numpy array of the shape (rows, count_fishes*4) and saves a csv file at a given path (path has to end on .csv).
    The information about each given fish (or object in general) should be first_position_x, first_position_y, second_position_x, second_position_y.
    It is assumed that the fish is looking into the direction of first_positon_x - second_position_x for x and first_positon_y - second_position_y for y.
    """
    if saveToFile:
        output = np.array([list(chain.from_iterable(("Fish_" + str(i) + "_linear_movement", "Fish_" + str(i) + "_angle_new_pos", "Fish_" + str(i) + "_angle_change_orientation") for i in range(0, int(np_array.shape[1]/4))))])
    else:
        output = np.empty([0,int(np_array.shape[1]/4)*3])

    for i in range(0, np_array.shape[0]-1):
        if i%1000 == 0:
            print("||| Frame " + str(i) + " finished. |||")
        new_row = [0 for k in range(0, int(3*np_array.shape[1]/4))]
        for j in range(0, int(np_array.shape[1]/4)):
            head_x = np_array[i, j*4]
            head_y = np_array[i, j*4+1]
            center_x = np_array[i, j*4+2]
            center_y = np_array[i, j*4+3]

            head_x_next = np_array[i+1, j*4]
            head_y_next = np_array[i+1, j*4+1]
            center_x_next = np_array[i+1, j*4+2]
            center_y_next = np_array[i+1, j*4+3]

            #look vector
            look_vector = (head_x - center_x, head_y - center_y)
            #new look vector
            look_vector_next = (head_x_next - center_x_next, head_y_next - center_y_next)
            #vector to new position
            vector_next = (head_x - head_x_next, head_y - head_y_next)

            new_row[j*3+1] = getAngle(look_vector, vector_next, mode = mode)
            new_row[j*3+2] = getAngle(look_vector, look_vector_next, mode = mode)
            temp = getDistance(head_x, head_y, head_x_next, head_y_next)
            #its forward movement if it's new position is not at the back of the fish and otherwise it is backward movement
            #if mode == "radians":
            new_row[j*3] = temp if new_row[j*2+1] > math.pi/2 and new_row[j*2+1] < 3/2*math.pi else -temp
            #else:
            #    new_row[j*3] = temp if new_row[j*2+1] > math.pi/2 and new_row[j*2+1] < 3/2*math.pi else -temp
        output = np.append(output, [new_row], axis = 0)

    if saveToFile:
        df = pd.DataFrame(data = output[1:], columns = output[0])
        df.to_csv(path_to_save_to, index = None, sep = ";")
    else:
        return output

def convertLocmotionToBin(path, path_to_save, clusters_path, probabilities = True):
    #get cluster centers
    clusters_mov, clusters_pos, clusters_ori = readClusters(clusters_path)

    #get locomotion
    df = pd.read_csv(path, sep = ";")
    loco = df.to_numpy()

    result = None
    #convert locomotion into bin representation for each fish
    for i in range(0, int(loco.shape[1]/3)):
        if probabilities:
            #compute distances to cluster centers and invert them (1/x)
            dist_mov = 1 / distancesToClusters(loco[:, i*3], clusters_mov)
            dist_pos = 1 / distancesToClusters(loco[:, i*3+1], clusters_pos)
            dist_ori = 1 / distancesToClusters(loco[:, i*3+2], clusters_ori)

            #get probabilites row-wise with softmax function and append header
            prob_mov = np.append(np.array([["Fish_" + str(i) + "_prob_mov_bin_" + str(j) for j in range(0, len(clusters_mov))]]), softmax(dist_mov), axis = 0)
            prob_pos = np.append(np.array([["Fish_" + str(i) + "_prob_pos_bin_" + str(j) for j in range(0, len(clusters_pos))]]), softmax(dist_pos), axis = 0)
            prob_ori = np.append(np.array([["Fish_" + str(i) + "_prob_ori_bin_" + str(j) for j in range(0, len(clusters_ori))]]), softmax(dist_ori), axis = 0)

            temp = np.append(np.append(prob_mov, prob_pos, axis = 1), prob_ori, axis = 1)
            if i == 0:
                result = temp
            else:
                result = np.append(result, temp , axis = 1)
        else:
            #todo
            pass

    #save it
    df = pd.DataFrame(data = result[1:], columns = result[0])
    df.to_csv(path_to_save, sep = ";")

def distancesToClusters(points, clusters):
    """
    computes distances from all points to all clusters
    not sure if this works for non 1d data
    """
    distances = None
    for j in range(0, len(clusters)):
        temp = np.abs(points - float(clusters[j])).reshape(-1, 1)
        if j == 0:
            distances = temp
        else:
            distances = np.append(distances, temp, axis = 1)

    return distances

def softmax(np_array):
    """
    Compute softmax values row-wise (probabilites)
    """
    temp = np.exp(np_array - np.max(np_array, axis = 1).reshape(-1, 1))
    return np.divide(temp, np.sum(temp, axis = 1).reshape(-1, 1))

def main():
    # file = "data/sleap_1_Diffgroup1-1.h5"

    # temp = extract_coordinates(file, [b'head', b'center'], fish_to_extract=[0,1,2])

    # #remove rows with Nans in it
    # temp = temp[~np.isnan(temp).any(axis=1)]

    # print("shape:",temp.shape)

    # #get locomotion and save it
    # getLocomotion(temp , "data/locomotion_data.csv")

    convertLocmotionToBin("data/locomotion_data.csv", "data/locomotion_data_bin.csv", "data/clusters.txt")

if __name__ == "__main__":
    main()