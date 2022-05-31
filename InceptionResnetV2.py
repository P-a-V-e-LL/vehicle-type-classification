import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import save_model

save_path = './models/'

def main():
    base_model = InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(160, 160, 3),
            pooling='avg'
        )
    x = base_model.output
    x = Dropout(1.0 - 0.8, name='Dropout')(x)

    # Bottleneck

    x = Dense(128, use_bias=False, name='Bottleneck')(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)
    #if self.use_l2:
    x = Lambda(lambda i: tf.math.l2_normalize(i, axis=1))(x)
    model = Model(base_model.input, x, name='inception_resnet_v2')
    model.save('./models/irv2.h5')
    #save_model(model, save_path)


if __name__ == '__main__':
    main()
