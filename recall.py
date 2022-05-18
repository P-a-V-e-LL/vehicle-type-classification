from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import argparse
import datetime

'''Отсутствует обработка new_individual, добавить после вычисления коэффицента близости'''

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_data",
        required=True,
        help="Path to .pickle base (train) data."
    )
    ap.add_argument(
        "--target_data",
        #required=True,
        default = False,
        help="Path to .pickle data to recall@1. Ignore if want to recall@1 train data."
    )
    ap.add_argument(
        "--tflite",
        #required=True,
        default = False,
        help="Set True if embeddings were obtained using tflite model."
    )
    ap.add_argument(
        "--filename",
        required=True,
        help="Metric's mistaces filename."
    )
    return vars(ap.parse_args())

def pickle_to_data(fname):
    f = open(fname, "rb")
    x = pickle.load(f)
    f.close()
    return x

def tflite_prep(data):
    '''
    При формировании структуры с опмощью tflite один вектор помещается в двойной массив,
    что не подходит для работы recall. Исправляем [[vector]] -> [vector]
    '''
    for i in range(len(data)):
        data[i] = data[i][0]
    return data

def class_encode(data):
    '''
    Кодирует все классы цифрами
    Пример: {0: 'lada', 1: 'reno'}
    '''
    classes = list(data.keys())
    encodings = {}
    for i in range(len(classes)):
        encodings[i] = classes[i]
    return encodings

def pickle_to_embeddings(data):
    '''
    Преобразовывает pickle файл с эмбеддингами в структуру
    входных данных для класса NearestNeighbors
    '''
    data_embeddings = []
    data_ids = []
    data_path = []
    for cl in list(data.keys()):
        for embedding in data[cl]:
            #print(len(data[cl]))
            data_embeddings.append(embedding['embedding'])
            data_ids.append(cl)
            data_path.append(embedding['path'])
    return data_embeddings, data_ids, data_path

def decode(data, decoder, target_paths, ignore=False):
    data = list(data)
    for i in range(len(data)):
        if ignore:
            data[i] = list(data[i])[1:]
        else:
            data[i] = list(data[i])
        for num in range(len(data[i])):
            #data[i][num] = decoder[data[i][num]]
            data[i][num] = {decoder[data[i][num]]: target_paths[i]}
    return data

def recall(data, target_ids, fname, n=1):
    rcl = 0
    f = open('./recall/'+fname+'_'+str(datetime.datetime.now())+'.txt', 'w')
    #print(len(data))
    for i in range(len(data)):
        target = target_ids[i]
        #result = data[i][0]
        result = list(data[i][0].keys())[0]
        if result == target:
            #print(result, target)
            rcl += 1
        else:
            f.write('Result: ' + result + ' | Target: ' + target + '\n')
            f.write(data[i][0][list(data[i][0].keys())[0]]+'\n')
            f.write('-'*100+'\n')
            print('Result: ' + result + ' Target: ' + target)
            print(data[i][0][list(data[i][0].keys())[0]])
    f.close()
    return rcl / len(data) * 100

def main():
    args = get_arguments()
    ignore_flag = False # флаг игнорирование первого ближайшего соседа

    if args['target_data']:
        target_data = args['target_data']
    else:
        target_data = args['base_data']
        ignore_flag = True

    #train_embeddings, train_ids = pickle_to_embeddings(pickle_to_data(args['base_data']))
    #val_embeddings, val_ids = pickle_to_embeddings(pickle_to_data(target_data))
    train_embeddings, train_ids, train_paths = pickle_to_embeddings(pickle_to_data(args['base_data']))
    val_embeddings, val_ids, val_paths = pickle_to_embeddings(pickle_to_data(target_data))
    encodings = class_encode(pickle_to_data(args['base_data']))

    if args['tflite']:
        train_embeddings = tflite_prep(train_embeddings)
        val_embeddings = tflite_prep(val_embeddings)

    if args['target_data']:
        target_ids = val_ids
        target_paths = val_paths
    else:
        target_ids = train_ids
        target_paths = train_paths

    neigh = NearestNeighbors(n_neighbors=5,metric='l2').fit(train_embeddings)
    val_nn_distances, val_nn_idxs = neigh.kneighbors(val_embeddings, 5, return_distance=True)
    #print(train_ids)
    #print(decode(val_nn_idxs[:10], target_ids, ignore_flag))

    print("RECALL@1", recall(decode(val_nn_idxs, train_ids, target_paths, ignore_flag), target_ids, args['filename']))


if __name__ == '__main__':
    main()
