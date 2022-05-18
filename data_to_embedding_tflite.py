import os
import sys
import pickle
import cv2
import numpy as np
import argparse
import tflite_runtime.interpreter as tflite

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "--root_dir",
        required=True,
        help="Path to data dir."
    )
    ap.add_argument(
        "--filename",
        default=False,
        help="New pickle data filename"
    )
    return vars(ap.parse_args())

def get_v(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    image = cv2.resize(image, (160,160))
    image = np.asarray(image)
    image = image.astype(np.float32)
    image /= 255
    samples = np.expand_dims(image, axis=0)
    samples = np.stack((samples.copy(),)*3, axis=-1)
    #yhat = model.predict(samples)
    #return yhat[0]
    return samples

def main():
    args = get_arguments()
    embeddings = {}
    interpreter = tflite.Interpreter(model_path=args['model_path'])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for cl in os.listdir(args['root_dir']):
        embeddings[cl] = []
        path = os.path.join(args['root_dir'], cl)
        for car in os.listdir(path):
            input_data = np.array(get_v(os.path.join(path, car)), dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            car_to_add = {'uid': car, 'path': os.path.join(path, car), 'embedding': output_data}
            embeddings[cl].append(car_to_add)
        print("{0} saved.".format(cl))

    f = open("./embedding_data/" + args['filename'] + ".pickle", "wb+")
    pickle.dump(embeddings, f)
    f.close()
    print("Данные сохранены!")


if __name__ == '__main__':
    main()
