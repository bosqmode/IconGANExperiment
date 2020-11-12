
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

class GAN:
    def __init__(self, d_lr = 0.0002, g_lr = 0.0002, training = True, input_shape = (64,64,3)):
        self.input_shape = input_shape
        self.training = training
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.initializer = RandomNormal(stddev=0.02)
        self.Discriminator = self.CreateDiscriminator()
        self.Generator = self.CreateGenerator()
        self.GAN = self.CreateGAN()
        self.GAN.summary()

    def Save(self, g_filename, d_filename):
        self.Generator.save_weights(g_filename)
        self.Discriminator.save_weights(d_filename)

    def Load(self, g_filename, d_filename):
        self.Generator.load_weights(g_filename)
        self.Discriminator.load_weights(d_filename)

    def CreateDiscriminator(self):
        edge_in = Input(shape=self.input_shape)
        target_in = Input(shape=self.input_shape)

        concat = Concatenate()([edge_in, target_in])
        x = Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer=self.initializer)(concat)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(512, (3,3), padding='same', kernel_initializer=self.initializer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(1,(3,3), padding='same', kernel_initializer=self.initializer)(x)
        out = Activation('sigmoid')(x)

        model = Model([edge_in, target_in], out)
        opt = Adam(lr=self.d_lr, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'], loss_weights=[0.5])
        return model
    
    def CreateGenerator(self):
        edge_in = Input(shape=self.input_shape)
        e1 = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(edge_in)
        e1 = LeakyReLU(alpha=0.2)(e1)
        e2 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(e1)
        e2 = BatchNormalization()(e2, training=self.training)
        e2 = LeakyReLU(alpha=0.2)(e2)
        e3 = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(e2)
        e3 = BatchNormalization()(e3, training=self.training)
        e3 = LeakyReLU(alpha=0.2)(e3)
        e4 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(e3)
        e4 = BatchNormalization()(e4, training=self.training)
        e4 = LeakyReLU(alpha=0.2)(e4)
        x = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(e4)
        x = Activation('relu')(x)
        d1 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(x)
        d1 = BatchNormalization()(d1, training=self.training)
        d1 = Dropout(0.4)(d1, training=self.training)
        d1 = Concatenate()([d1, e4])
        d1 = Activation('relu')(d1)
        d2 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(d1)
        d2 = BatchNormalization()(d2, training=self.training)
        d2 = Dropout(0.4)(d2, training=self.training)
        d2 = Concatenate()([d2, e3])
        d2 = Activation('relu')(d2)
        d3 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(d2)
        d3 = BatchNormalization()(d3, training=self.training)
        d3 = Concatenate()([d3, e2])
        d3 = Activation('relu')(d3)
        d4 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(d3)
        d4 = BatchNormalization()(d4, training=self.training)
        d4 = Concatenate()([d4, e1])
        d4 = Activation('relu')(d4)
        g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=self.initializer)(d4)
        out = Activation('tanh')(g)

        model = Model(edge_in, out)
        return model

    def CreateGAN(self):
        self.Discriminator.trainable = False
        edge_in = Input(shape=self.input_shape)
        g_out = self.Generator(edge_in)
        d_out = self.Discriminator([edge_in, g_out])
        model = Model(edge_in, [d_out, g_out])
        model.summary()
        opt = Adam(learning_rate=self.g_lr, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[10,80])
        return model