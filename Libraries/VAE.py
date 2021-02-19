#TF libraries

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense 
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda,Conv2D,Conv2DTranspose,LeakyReLU,ReLU
from keras.initializers import Constant
from keras.models import Model
from keras.optimizers import Adam
from keras import initializers
import keras.backend as K

batch_size = 32
image_size=64
latent_dim = 64
start_lr = 1e-6

#Initialize model
def Initialize_VAE(beta_1=0.5,beta_2=0.999,Loss_type='Original',Beta=10**-2,free_bits=40):
    with tf.device('/device:GPU:0'):
        models= create_vae()
        custom_loss=Loss(models,Loss_type,Beta,free_bits)
        models["vae"].compile(optimizer=Adam(learning_rate=start_lr, beta_1=beta_1, beta_2=beta_2), loss=custom_loss)
        return models

#Encoder
def encoder_Article(input_img):
    #He initialization for Relu activated layers
    #Relu halfs the variance, He doubles it
    Heinitializer = initializers.HeNormal()

    x = Conv2D(filters=64, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(input_img)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=128, kernel_size=4,strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=256, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=512, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=4096, kernel_size=4, strides=1,padding='valid',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    z_mean = Dense(latent_dim,)(x)
    z_log_var = Dense(latent_dim, )(x)

    return z_mean,z_log_var

#Decoder
def decoder_Article(z):
    #He initialization for Relu activated layers
    #Relu halfs the variance, He doubles it
    Heinitializer = initializers.HeNormal()

    #Xavier intialization for differentiable functions activated layers
    #Xavier keeps the variance the same
    Xavierinitializer=initializers.GlorotNormal()

    x = Reshape(target_shape=(1, 1, 64))(z)

    x = Conv2DTranspose(filters=512, kernel_size=4, strides=1,padding='valid',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(filters=256, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(filters=128, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(filters=64, kernel_size=4, strides=2,padding='same',use_bias=False,kernel_initializer=Heinitializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    decoded = Conv2DTranspose(filters=1, kernel_size=4, strides=2,padding='same',
                        activation='sigmoid',use_bias=False,kernel_initializer=Xavierinitializer)(x)


    return decoded

#A function to turn keras tensor into numpy representation
def tensor_values(x):
  return K.eval(x)

#Reparametrization trick in latent space
def reparameterize(args):
    mean,logvar=args
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar/2) + mean

#VAE
def create_vae():
    models = {}

    #Encoder
    input_img = Input(batch_shape=(batch_size, image_size, image_size, 1))
    z_mean, z_log_var=encoder_Article(input_img)

    #Reparametrization
    l=Lambda(reparameterize, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    #Models to get latent variables
    models["encoder"]  = Model(input_img, l, name='Encoder')
    models["z_meaner"] = Model(input_img, z_mean, name='Enc_z_mean')
    models["z_log_varer"] = Model(input_img, z_log_var, name='Enc_z_log_var')

    #Decoder
    z = Input(shape=(latent_dim, ))
    decoded=decoder_Article(z)
    models["decoder"] = Model(z, decoded, name='Decoder')

    #Complete VAE
    models["vae"]     = Model(input_img, models["decoder"](models["encoder"](input_img)), name="VAE")

    return models

#Losses
def Loss(models,Loss_type='Original',Beta=10**-2,free_bits=40):
    def custom_loss(x,decoded):

        #Reconstruction quality
        flattened_x=K.reshape(x,shape=(batch_size,image_size*image_size))
        flattened_decoded=K.reshape(decoded,shape=(batch_size,image_size*image_size))
        Log_loss=image_size*image_size*tf.keras.losses.binary_crossentropy(flattened_x,flattened_decoded)

        #Latent space regularization
        mean = models['z_meaner'](x)
        logvar=models['z_log_varer'](x)
        KL_loss=0.5 * K.sum(1 + logvar - K.square(mean) - K.exp(logvar), axis=-1)

        #Choice of loss function
        if Loss_type=='Original':
            return (Log_loss-KL_loss)/image_size/image_size

        elif Loss_type=='Beta':
            return (Log_loss-Beta*KL_loss)/image_size/image_size

        elif Loss_type=='Flow':
            free_bits_tensor=K.constant(free_bits*np.ones(KL_loss.shape[0]))
            return (Log_loss-K.maximum(-free_bits_tensor,KL_loss))/image_size/image_size

    return custom_loss

#Callbacks for training
import ast
import seaborn as sns
import matplotlib.pyplot as plt

def plot_galaxies(*args,save_filename=False):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    figure = np.zeros((image_size * len(args), image_size * n))

    #Each argument is a separate row 
    for i in range(n):
        for j in range(len(args)):
            figure[j * image_size: (j + 1) * image_size,
                   i * image_size: (i + 1) * image_size] = args[j][i].squeeze()


    plt.figure(figsize=(2*n, 2*len(args)))
    plt.imshow(figure, cmap='Greys_r')
    plt.grid(False)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Save figure
    if save_filename:
        plt.savefig(save_filename)

    plt.show()



def Show_latent_distr(models,x_test):
  z_mean=tensor_values(models['z_meaner'].predict(x_test))
  z_log_var=tensor_values(models['z_log_varer'].predict(x_test))

  plt.hist(np.mean(z_mean,axis=0),alpha=0.5,label='mean')
  plt.hist(np.mean(z_log_var,axis=0),alpha=0.5,label='log_var')
  plt.title('Latent space distribution')
  plt.legend()

  plt.tight_layout()
  plt.show()

def Learning_curve(filename,start_epoch=0,stop_epoch=1000,save_filename=False):
  logs_file=open(filename)
  lines=logs_file.readlines()
  logs_file.close()

  loss=np.array([])
  val_loss=np.array([])
  for line in lines:
    note=ast.literal_eval(line)
    loss=np.append(loss,[note['loss']])
    val_loss=np.append(val_loss,[note['val_loss']])

  start_index=start_epoch//10
  stop_index=np.minimum(len(loss),stop_epoch//10+1)
  plt.plot(10*np.arange(start_index,stop_index),loss[start_index:stop_index],label='Train')
  plt.plot(10*np.arange(start_index,stop_index),val_loss[start_index:stop_index],label='Validation')
  plt.ylabel('Loss')
  plt.xlabel('epoch number')
  plt.title('Learning curve')
  plt.legend()

  #Save figure
  if save_filename:
    plt.savefig(save_filename)

  plt.show()
