from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import pandas as pd
from tqdm.auto import tqdm
import argparse

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train_data",
        #required=True,
        help="Path to .pickle train data."
    )
    ap.add_argument(
        "--val_data",
        #required=True,
        help="Path to .pickle val data."
    )
    return vars(ap.parse_args())

def pickle_to_data(fname):
    f = open(fname, "rb")
    x = pickle.load(f)
    f.close()
    return x

def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def get_predictions(test_df,threshold=0.2):
    predictions = {}
    for i,row in tqdm(test_df.iterrows()):
        if row.image in predictions:
            if len(predictions[row.image])==5:
                continue
            predictions[row.image].append(row.target)
        #elif row.confidence>threshold:
        elif row.confidence < threshold:
            predictions[row.image] = [row.target,'new_individual']
        else:
            predictions[row.image] = ['new_individual',row.target]

    #for x in tqdm(predictions):
    #    if len(predictions[x])<5:
    #        remaining = [y for y in sample_list if y not in predictions]
    #        predictions[x] = predictions[x]+remaining
    #        predictions[x] = predictions[x][:5]

    return predictions

def pickle_to_embeddings(data):
    '''
    Преобразовывает pickle файл с эмбеддингами в структуру
    входных данных для класса NearestNeighbors
    '''
    data_embeddings = []
    data_ids = []
    data_targets = []
    for cl in list(data.keys()):
        for embedding in data[cl]:
            data_embeddings.append(embedding['embedding'])
            data_targets.append(cl)
            data_ids.append(embedding['path'])
    return data_embeddings, data_ids, data_targets


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

def class_encode_all(train_data, val_data):
    '''
    Кодирует все классы цифрами
    Пример: {0: 'lada', 1: 'reno'}
    '''
    classes = list(train_data.keys())
    for cl in val_data.keys():
        if cl not in classes:
            classes.append(cl)
    encodings = {}
    for i in range(len(classes)):
        encodings[i] = classes[i]

    return encodings

def targets_fix(targets, coder):
    '''Названия классов в числа.'''
    for i in range(len(targets)):
        for key, value in coder.items():
            if value == targets[i]:
                code = key
        targets[i] = code
    return np.asarray(targets)

def main():
    args = get_arguments()
    best_threshold_adjusted = 0.6

    train_embeddings, train_ids, train_targets = pickle_to_embeddings(pickle_to_data(args['train_data']))
    val_embeddings, val_ids, val_targets = pickle_to_embeddings(pickle_to_data(args['val_data']))
    target_encodings = class_encode_all(pickle_to_data(args['train_data']), pickle_to_data(args['val_data']))

    train_targets = targets_fix(train_targets, target_encodings)
    val_targets = targets_fix(val_targets, target_encodings)

    neigh = NearestNeighbors(n_neighbors=100,metric='l2')
    neigh.fit(train_embeddings)
    val_nn_distances, val_nn_idxs = neigh.kneighbors(val_embeddings, 100, return_distance=True)
    allowed_targets = set([target_encodings[x] for x in np.unique(train_targets)])

    val_targets_df = pd.DataFrame(np.stack([val_ids,val_targets],axis=1),columns=['image','target'])
    val_targets_df['target'] = val_targets_df['target'].astype(int).map(target_encodings)
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets),'target'] = 'new_individual'
    val_targets_df.target.value_counts()
    val_df = []

    for i in tqdm(range(len(val_ids))):
        id_ = val_ids[i]
        targets = train_targets[val_nn_idxs[i]]
        distances = val_nn_distances[i]
        subset_preds = pd.DataFrame(np.stack([targets,distances],axis=1),columns=['target','distances'])
        subset_preds['image'] = id_
        val_df.append(subset_preds)

    val_df = pd.concat(val_df).reset_index(drop=True)
    val_df['confidence'] = val_df['distances']
    #val_df['confidence'] = 1-val_df['distances']
    val_df = val_df.groupby(['image','target']).confidence.max().reset_index()
    val_df = val_df.sort_values('confidence',ascending=True).reset_index(drop=True)
    #val_df = val_df.sort_values('confidence',ascending=False).reset_index(drop=True)
    val_df['target'] = val_df['target'].map(target_encodings)
    #val_df.to_csv('val_neighbors.csv')
    val_df.image.value_counts().value_counts()

    ## Compute CV
    #th = 1.1
    cv = 0
    ths = [x/10 for x in range(1, 40)]

    for th in ths:
        all_preds = get_predictions(val_df,threshold=th)
        for i,row in val_targets_df.iterrows():
            target = row.target
            preds = all_preds[row.image]
            val_targets_df.loc[i,th] = map_per_image(target,preds)

        cv = val_targets_df[th].mean()

        #print(f"CV at threshold {th}: {cv}")
        val_targets_df.describe()

        ## Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
        val_targets_df['is_new_individual'] = val_targets_df.target=='new_individual'
        #print(val_targets_df.is_new_individual.value_counts().to_dict())
        val_scores = val_targets_df.groupby('is_new_individual').mean().T
        val_scores['adjusted_cv'] = val_scores[True]*0.38+val_scores[False]*0.62 # коэффиценты меняются в зависимости от количества данных
        #val_scores['adjusted_cv'] = val_scores[True]*0.1+val_scores[False]*0.9
        print(val_scores)
        #best_threshold_adjusted = val_scores['adjusted_cv'].idxmax()

        best_threshold_adjusted = val_scores['adjusted_cv'].idxmax()
        print("best_threshold",best_threshold_adjusted)
        val_scores

        train_embeddings = np.concatenate([train_embeddings,val_embeddings])
        train_targets = np.concatenate([train_targets,val_targets])
        #print(train_embeddings.shape,train_targets.shape)
        print("-"*50)


if __name__ == '__main__':
    main()
