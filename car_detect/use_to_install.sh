
#!/bin/bash

#TEMP_DIRNAME="car_detect"
#mkdir ${TEMP_DIRNAME}
#cd ${TEMP_DIRNAME}
mc cp minio-inventos/production-net/anpr/ru_one_linear/20211224/anpr_ru_one_linear_20211224.tflite .
mc cp minio-inventos/production-net/anpr/ru_one_linear/20211224/alphabet.txt .
mc cp minio-inventos/neuroweb/networks/yolov4_csp/onnx/yolov4-csp.onnx .
mc cp minio-inventos/neuroweb/networks/yolov4_csp/onnx/yolov4_2_3_512_512_static.onnx .
