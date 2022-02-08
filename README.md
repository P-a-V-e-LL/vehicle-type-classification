

# Vehicle Type Classification

Здесь находятся программы, предназначенные для сборки датасета, обучения нейронной сети и оценки ее качества.

# Installation
1.Клонировать репозиторий:
```
git clone git@gitlab.inventos.ru:neurowebs/vehicle-type-classification-2.git
```
2.Создать новое вируальное окружение:
```
python3.7 -m venv new_venv
source new_venv/bin/activate
```
3.Установить зависимости:
```
pip install -r requirements.txt
```
4.Перейти в папку car_detect. Использовать скрипт use_to_install.sh

# car_nn.py

Программа вырезает кабины автомобилей из видео.
Каждую вырезанную кабину называет в соответствии с номером автомобиля.
Использует нейросеть распознавания номеров, распознавания классов автомобилей.
Используем для формирования первоначального (неразмеченного) датасета.

Для работы подгатавливаются папка с видео, на которых необходимо распознать автомобили, папка для сохранения изображений распознанных автомобилей.
Также необходимо указать пути для файлов: нейронная сеть для распознавания номеров, нейронная сеть для распознавания автомобилей, текстовый файл с описанием классов авто.
Три последних файла можно найти в папке car_detect.
Пути до вышеописанных файлов и папок указываются внутри программы.

# model_training.py

Программа выполняет обучение нейронной сети с помощью tensorflow. Результатом работы программы является модель формата .h5, обученная на подготовленных наборах данных.

Для работы программы необходимо подготовить два набора данных - обучающий и валидационный. Каждый набор должен быть разделен по классам и иметь вид:

```
root_dir:
	Class_1:
		img1
		img2
		img3
		...
	Class_2:
		img1
		img2
		img3
		...
	Class N:
		...
```

Все изображения должны быть в чёрно-белом формате (для этого можно использовать программу to_gray.py).
root_dir является папкой, в которой хранится весь набор данных. Именно она указывается как путь к набору данных.
Также необходимо указать путь для сохранения обученной модели.
Во время обучения программа сохраняет значения ошибки и точности в папку logs. По этим данным можно построить графики с помощью tensorboard.

Далее необходимо указать путь до модели нейронной сети, котрая будет обучаться в процессе работы программы. Модель должна быть в формате .h5.
И под конец необходимо указать такие параметры, как размер батча, ширину и высоту изображения для предобработки, количество эпох обучения, а текже колчичество одучающих и валидационных изображений (для этого можно воспользоваться программной check.py).

Путь до вышеописанных папок и файлов указывается внутри программы, числовые переменные задаются там же.

# check.py

Выводит общее количество изображений набора данных.

Для работы нужно указать root_dir для набора данных.

# ds_train_val_test_cut.py

Проводит разбиение одного целого набора данных на обучающий и валидационный в соотношении 8/2.

Для работы программы необходимо указать путь до root_dir исходного набора данных (для разделения), а также папки для сохранения обучающей и валидационной выборок. Исходный набор данных не будет изменен в процессе работы программы.

# GRZ_get.py

Программа позволяет, используя государственный номер транспортного средства, узнать марку, модель и год транспортного средства. Программа обращается за данными к SpectrumData API.

Для работы программы необходима версия python 3.9 и выше. Данную версию нужно установить отдельно - только этой программе требуется высокая версия python.
Перед запуском программы необходимо получить токен авторизации на SpectrumData и указать его в программе. Помимо этого, на вход программа принимает путь до папки с изображениями, в названии которых находится лишь номер транспортного средства (прим. 'E020BX154.jpg').

В процессе работы программы каждое изображение получает новое название в соответствии со своими маркой и моделью, либо удаляется (в случае отсутствия номера в базе данных). Каждый новый номер записывается в файл data.txt, таким образом, если какой-то номер уже когда-то был распознан, то название будет просто извлечено из файла без обращения к SpectrumData.

# to_gray.py

Преобразует изображения для обучающей и валидационной выборок в чёрно-белый формат.

Для работы программы необходимо указать папку, в которой находятся необходимые наборы данных. Папки с наборами данных должны называться train и val соответственно, и располагаться внутри указанной ранее папки.

# affin.py

Преобразует изображения используя случайное сочитание эффектов (поворот, отзеркаливание, шум, контраст, дождб, снег, туман, размытие при движении, зум и обрезание небольшой части изображения).

Для работы программы необходимо указать root_dir для набора данных. В результате работы программы в папке каждого класса появится n новых изображений с эффектами. Каждое ново изображение основано на уже существующем внутри класса изображении. Также можно внутри программы указать общее желаемое число изображений в каждом классе (по умолчанию 210).

Все новые изображения сохраняются в папку класса, в котором лежали их оригиналы.

# recognize_test.py

Не доделан! Должен использоваться как классификатор для существующих классов. В будущем должен использоваться для отбора изображений без обращения к SpectrumData в паре с переобученной моделью.

# vizualize_cluster.py

Визуализирует кластеризацию модели. Для работы программы нужна модель .h5 и набор размеченных данных, пути до которых указываются внутри программы.

В конце работы программы требуется ввести с клавиатуры название файла, который будет сохранен как изображение с построенным графиком.

# Папка car_detect
Содержит следующие файлы:
- alphabet.txt - буквы и цифры для обозначения номеров.
- anpr_ru_one_linear_20210426.tflite - нейронная сеть для распознавания номеров авто.
- names.txt - названия классов, определяемые на каждом авто.
- tflite_runtime-2.5.0-cp37-cp37m-linux_x86_64.whl - билд библиотеки tflite_runtime.
- yolov4_2_3_512_512_static.onnx - нейронная сеть для определения авто и его классов.
- yolov4-csp.onnx - то же самое, что и (5), но недоделанная с измененными значениями входов.

# Наборы данных

- cars196 - размеченный набор данных из открытого доступа.
- dataset3.11 - вручную собранный, размеченный набор данных.

Наборы данных можно молучить с помощью скрипта from_minio.sh. Необходимо запустить скрипт с единственным аргументом - названием датасета (указаны выше).
```
bash from_minio.sh dataset3.11
```
# Модели

- facenet_keras - оригинальный Facenet.

Модели можно получить с помощью скрипта get_model_from_minio.sh. Необходимо запустить скрипт с единственным аргументом - названием модели (указаны выше). Выбранная модель будет загружена в папку models.

```
bash get_model_from_minio.sh facenet_keras
```
