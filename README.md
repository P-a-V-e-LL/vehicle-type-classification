

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

Обязательные аргументы:
- --video_path - путь к папке с видео для распознавания.

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

Обязательные аргументы:
- --model_path - путь до .h5 keras модели.
- --train_data - путь до папки с обучающей выборкой данных.
- --val_data - путь до папки с валидационной выборкой данных.
- --epochs - количество эпох обучения.

Необязательные аругменты:
- --batch_size - размер батча (128 по умолчанию).

# ds_train_val_test_cut.py

Проводит разбиение одного целого набора данных на обучающий и валидационный в заданном соотношении.

Для работы программы необходимо указать путь до root_dir исходного набора данных (для разделения), а также папки для сохранения обучающей и валидационной выборок. Исходный набор данных не будет изменен в процессе работы программы.

Обязательные аргументы:
- --data_dir - путь до общего набора данных.
- --train_dir - путь для сохранения обучающей выборки данных.
- --val_dir - путь до сохранения валидационной выборки данных.


# GRZ_get.py

Программа позволяет, используя государственный номер транспортного средства, узнать марку, модель и год транспортного средства. Программа обращается за данными к SpectrumData API.

Для работы программы необходима версия python 3.9 и выше. Данную версию нужно установить отдельно - только этой программе требуется высокая версия python.
Перед запуском программы необходимо получить токен авторизации на SpectrumData и указать его в программе. Помимо этого, на вход программа принимает путь до папки с изображениями, в названии которых находится лишь номер транспортного средства (прим. 'E020BX154.jpg').

В процессе работы программы каждое изображение получает новое название в соответствии со своими маркой и моделью, либо удаляется (в случае отсутствия номера в базе данных). Каждый новый номер записывается в файл data.txt, таким образом, если какой-то номер уже когда-то был распознан, то название будет просто извлечено из файла без обращения к SpectrumData.

# to_gray.py

Преобразует изображения для обучающей и валидационной выборок в чёрно-белый формат.

Для работы программы необходимо указать папку, в которой находятся необходимые наборы данных. Папки с наборами данных должны называться train и val соответственно, и располагаться внутри указанной ранее папки.

Обязательные аргументы:
- --root_dir - путь до набора данных.

# affin.py

Преобразует изображения используя случайное сочитание эффектов (поворот, отзеркаливание, шум, контраст, дождб, снег, туман, размытие при движении, зум и обрезание небольшой части изображения).

Для работы программы необходимо указать root_dir для набора данных. В результате работы программы в папке каждого класса появится n новых изображений с эффектами. Каждое ново изображение основано на уже существующем внутри класса изображении. Также можно внутри программы указать общее желаемое число изображений в каждом классе (по умолчанию 210).

Все новые изображения сохраняются в папку класса, в котором лежали их оригиналы.

Обязательные аргументы:
- --root_dir - путь к выборке данных.

# recognize_test.py

Не доделан! Должен использоваться как классификатор для существующих классов. В будущем должен использоваться для отбора изображений без обращения к SpectrumData в паре с переобученной моделью.

# vizualize_cluster.py

Визуализирует кластеризацию модели. Для работы программы нужна модель .h5 и набор размеченных данных, пути до которых указываются внутри программы.

Обязательные аргументы:
- --model_path - путь до .h5 keras модели.
- --root_dir - путь до папки с выборкой данных.
- --filename - название файла, в котором будут сохранены данные.

# data_to_embedding.py

Сохраняет эмбединги выборки данных в файл с расширением .pickle. Для работы программы нужны обученная модель .h5 и выборка данных для сохранения.

После работы программы файл с данными будет сохранен в папке embedding_data.

Обязательные аргументы:
- --model_path - путь до .h5 keras модели.
- --root_dir - путь до папки с выборкой данных.
- --data - название файла для сохранения данных.

Аргументы на выбор (только один):
- --all - сохраняет все доступыне изображения (False по умолчанию).
- --n - количество изображений для сохранения из каждого класса (1 по умолчанию).


# get_knn.py

Вычисляет принадлежность тестового изображения к классам, записанным в .pickle файл. Для работы программы нужна обученная модель, тестовое изображение и файл с эмбеддингами и их классами.

Обязательные аргументы:
- --model_path - путь до .h5 keras модели.
- --test_image_path -  путь до изображение, которое нужно классифицировать.
- --pickle_file_path - путь до .pickle файла с данными.

Опциональные аргументы:
- --min_dist - минимильное расстояние между классами (0.6 по умолчанию).
- --knn_count - количество ближейших соседей, которое нужно вывести (10 по умолчанию).

# pickle_to_tensorboard.py

Переводит данные из .pickle файла в формат для визуализации в [embedding projector](https://projector.tensorflow.org/). На выходе программы получаются дв файла: вектора и метаданные.

Обязательные аргументы:
- --pickle_path - путь до .pickle файла с индексом эмбеддингов.

# data_to_csv.py

Записывает информацию о датасете в csv файл.
Структура записи: [название изображения, название класса, высота, ширина, выборка, путь до изображения]

Обязательные аргументы:
- --filename - название файла для сохранения (будет расположен в папке ./data/).
- --data_path - путь к набору данных (местоположение выборок данных):

Необязательные аргументы:
- train_data - флаг наличия обучающей выборки в директории data_path;
- val_data - флаг наличия валидационной выборки в директории data_path;
- test_data - флаг наличия тестовой выборки в директории data_path.

# change_emb_layer.py

Меняет размер выходного вектора модели facenet.

Обязательные аргументы:
- --model_name - название для новой модели. Новая модель будет сохранена в папке ./models/
- --dim - размер для выходного вектора.

Необязательыне аргументы:
- --l2 - добавляет l2 нормализацию в конец модели если True, по умолчанию False.

# Наборы данных

- cars196 - размеченный набор данных из открытого доступа.
- dataset3.11 - вручную собранный, размеченный набор данных.

Наборы данных можно молучить с помощью скрипта from_minio.sh. Необходимо запустить скрипт с единственным аргументом - названием датасета (указаны выше).
```
bash from_minio.sh dataset3.11
```
# Модели

- facenet_keras - оригинальный Facenet с выходным вектором 128 D.
- facenet512 - оригинальный Facenet с выходным вектором 512 D.

Модели можно получить с помощью скрипта get_model_from_minio.sh. Необходимо запустить скрипт с единственным аргументом - названием модели (указаны выше). Выбранная модель будет загружена в папку models.

```
bash get_model_from_minio.sh facenet_keras
```
