import tensorflow as tf
import numpy as np

def generator(project_shape, filters_list, name="generator"):
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.layers.Dense(
        units=np.prod(project_shape),
        input_shape=[100],
        use_bias=False,  
        kernel_initializer='glorot_normal'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape(target_shape=project_shape))
    for filters in filters_list[:-1]:
        model.add(tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size=[4,4,4],
            strides=[2,2,2],
            padding="same",
            use_bias=False,
            kernel_initializer='glorot_normal'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv3DTranspose(
        filters=filters_list[-1],
        kernel_size=[4,4,4],
        strides=[1,1,1],
        padding="same",
        activation=tf.nn.tanh,
        kernel_initializer='glorot_normal'
    ))

    return model


def discriminator(filters_list, name="discriminator"):
    model = tf.keras.Sequential(name=name)
    model.add(tf.keras.Input(shape=[32,32,32,1]))
    for filters in filters_list:
        model.add(tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=[4, 4, 4],
            strides=[2,2,2],
            padding="same",
            bias_initializer='zeros',
            kernel_initializer='glorot_normal'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=1,
        activation=tf.nn.sigmoid,
        kernel_initializer='glorot_normal'
    ))

    return model


class ThreeDGAN(object):
    def __init__(
        self,
        project_shape,
        gen_filters_list,
        disc_filters_list
        ):
        self.project_shape = project_shape
        self.gen_filters_list = gen_filters_list
        self.disc_filters_list = disc_filters_list

        self.generator = generator(self.project_shape,self.gen_filters_list)
        self.discriminator = discriminator(self.disc_filters_list)
    
    def generator_loss(self, z):
        x_fake = self.generator(z, training=True)
        fake_score = self.discriminator(x_fake, training=True)

        loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(fake_score), y_pred=fake_score, from_logits=False
        )

        return loss
        
    def discriminator_loss(self, x, z):
        x_fake = self.generator(z, training=True)
        fake_score = self.discriminator(x_fake, training=True)
        true_score = self.discriminator(x, training=True)

        loss = tf.keras.losses.binary_crossentropy(
                y_true=tf.ones_like(true_score), y_pred=true_score, from_logits=False 
                ) + tf.keras.losses.binary_crossentropy(
                y_true=tf.zeros_like(fake_score), y_pred=fake_score, from_logits=False
            )
        
        return loss