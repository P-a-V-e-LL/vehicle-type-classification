import os
import argparse
#import tensorflow as tf
import cv2
import numpy as np
#import tflite_runtime.interpreter as tflite
from classifier import Classifier
#import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

'''Описание коэффицента близости таким способом более не используется ввиду
неинформативности.'''

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "-pfp",
        "--pickle_file_path",
        required=True,
        help="Path to .pickle data file."
    )
    return vars(ap.parse_args())

def image_to_np(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    return image

def get_distance(embedding_1, embedding_2):
    '''Вычисляет расстояние между двумя векторами.'''
    return np.linalg.norm(embedding_1 - embedding_2)

def average(lst):
    return sum(lst) / len(lst)

def main():
    args = get_arguments()
    classifier1 = Classifier(args['model_path'], args['pickle_file_path'], 6)
    classifier2 = Classifier(args['model_path'], args['pickle_file_path'], 6)
    true_list =[]
    false_list = []
    for car1 in classifier1.pickle_data.keys():
        for embedding1 in classifier1.pickle_data[car1]:
            for car2 in classifier2.pickle_data.keys():
                for embedding2 in classifier2.pickle_data[car2]:
                    distance = float("{:.3f}".format(get_distance(embedding1, embedding2)))
                    #print(car1 == car2 and distance != 0.)
                    print(car1, car2)
                    if distance != 0.0:
                        if car1 == car2:
                            true_list.append(distance)
                        else:
                            false_list.append(distance)
                    else:
                        print(distance)

    print("-"*50)
    print(average(true_list), min(true_list), max(true_list))
    print("-"*50)
    print(average(false_list), min(false_list), max(false_list))

    fig = plt.hist(true_list, bins = 1000)
    filepath = './cluster_images/1full.png'
    #filepath1 = './cluster_images/2cut_true.png'
    #plt.savefig(filepath1, dpi=300)

    fig = plt.hist(false_list, bins = 1000)
    #filepath2 = './cluster_images/2cut_false.png'
    plt.savefig(filepath, dpi=300)
    #plt.show()

main()
