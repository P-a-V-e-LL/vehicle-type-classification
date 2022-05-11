import os
import pickle
import csv
import argparse
import imagesize
import sys

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_path",
        #required=True,
        help="Data path."
    )
    ap.add_argument(
        "--filename",
        required=True,
        help="Filename to new csv file."
    )
    ap.add_argument(
        "--train_data",
        default=False,
        help="Train data flag (if exists)."
    )
    ap.add_argument(
        "--val_data",
        default=False,
        help="Val data flag (if exists)."
    )
    ap.add_argument(
        "--test_data",
        default=False,
        help="Test data flag (if exists)"
    )
    ap.add_argument(
        "--all_clear_data",
        default=False,
        help="Clear data flag (if exists)"
    )
    return vars(ap.parse_args())

def write_csv(filename, samples):
    '''
    Записывает данные в csv файл.
    Аргументы:
        filename - название файла для записи, будет располагаться в папке ./data/;
        samples - dict, содержащий данные в формате {'выборка данных': 'путь до выборки'} (прим. {'train': './data/train', })
    '''
    f = open("./data/" + filename + ".csv", "w+", encoding="UTF8")
    writer = csv.writer(f)
    writer.writerow(["image_name", "class_name", "height", "width", "sample", "image_path", "view"]) # view

    for sample in samples.keys():
        classes = os.listdir(samples[sample])
        classes_count = len(classes)
        for cl in classes:
            cl_path = os.path.join(samples[sample], cl)
            #cars = os.listdir(cl_path)
            front_cl_path = os.path.join(cl_path, "front")
            cars = os.listdir(front_cl_path)
            print(cars)
            cars_count = len(cars)
            i = 0
            for car in cars:
                car_path = os.path.join(front_cl_path, car)
                width, height = imagesize.get(car_path)
                writer.writerow([car, cl, height, width, sample, car_path, "front"])
                i+=1
                sys.stdout.write('\r' + ' '*50 + '\r')
                sys.stdout.write("{} - {}/{}".format(cl, i, cars_count))
                sys.stdout.flush()
            #cars = os.listdir(cl_path)
            back_cl_path = os.path.join(cl_path, "back")
            cars = os.listdir(back_cl_path)
            cars_count = len(cars)
            i = 0
            for car in cars:
                car_path = os.path.join(back_cl_path, car)
                width, height = imagesize.get(car_path)
                writer.writerow([car, cl, height, width, sample, car_path, "back"])
                i+=1
                sys.stdout.write('\r' + ' '*50 + '\r')
                sys.stdout.write("{} - {}/{}".format(cl, i, cars_count))
                sys.stdout.flush()
    f.close()

def main():
    args = get_arguments()
    samples = {}

    if args["train_data"]:
        samples["train"] = os.path.join(args["data_path"], "train/")
    if args["val_data"]:
        samples["val"] = os.path.join(args["data_path"], "val/")
    if args["test_data"]:
        samples["test"] = os.path.join(args["data_path"], "test/")
    if args["all_clear_data"]:
        samples["clear"] = os.path.join(args["data_path"], "clear/")

    write_csv(args['filename'], samples)


if __name__ == '__main__':
    main()
