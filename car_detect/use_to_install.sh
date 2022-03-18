
#!/bin/bash

#TEMP_DIRNAME="car_detect"
#mkdir ${TEMP_DIRNAME}
#cd ${TEMP_DIRNAME}
mc cp minio-inventos/production-net/anpr/ru_one_linear/20211224/anpr_ru_one_linear_20211224.tflite .
mc cp minio-inventos/production-net/anpr/ru_one_linear/20211224/alphabet.txt .
mc cp minio-inventos/neuroweb/networks/yolox/onnx/yolox_s_20220119/five_car_class_yolox_s_190122.onnx .