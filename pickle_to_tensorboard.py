import os
import tensorflow as tf
import pickle
import numpy as np
import argparse

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pickle_path",
        required=True,
        help="Path to .pickle data file."
    )
    return vars(ap.parse_args())

#print(np.array2string(embeddings['LADA VAZ 2111'][0])[1:-1][-1])

def main():
    args = get_arguments()
    ff = open(args['pickle_path'], "rb")
    embeddings = pickle.load(ff)
    ff.close()
    f = open('./logs/vectors.tsv', "w+")
    f_w = open('./logs/v_metadata.tsv', "w+")
    for emb in embeddings.keys():
      for mas in embeddings[emb]:
          f_w.write("{}\n".format(emb))
          for i in range(len(mas)):
              f.write("{}\t".format(mas[i]))
          f.write("\n")
    f.close()
    f_w.close()


if __name__ == '__main__':
    main()
