
#!/bin/bash

TEMP_DIRNAME="data"

mkdir ${TEMP_DIRNAME} 
cd ${TEMP_DIRNAME}
mkdir ${1} 
cd ${1}
mc cp minio-inventos/datasets/fit/car_and_lic/vehicle-type-classification-2/${1}.tar.gz .
tar -xzf ${1}.tar.gz