
#!/bin/bash

TEMP_DIRNAME="temp"

mkdir ${TEMP_DIRNAME} 
cd ${TEMP_DIRNAME}
mc cp minio-inventos/datasets/fit/car_and_lic/vehicle-type-classification-2/${1}.tar.gz .
