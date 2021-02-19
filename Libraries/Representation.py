import ast
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras.backend as K
import copy

#Functions for training process callbacks
def plot_galaxies(*args,save_filename=False):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    image_size=args[0].shape[1]
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


#Study of trained models properties
def present_reconstruction(models,images_to_reconstruct,resid=False):
    decoded_to_reconstruct=models['vae'].predict(images_to_reconstruct)
    if resid:
      residuals=decoded_to_reconstruct-images_for_reconst
      plot_galaxies(images_for_reconst[:10],decoded_to_reconstruct[:10],residuals[:10])
      plt.colorbar()
    else:
      plot_galaxies(images_for_reconst[:10],decoded_to_reconstruct[:10])

#Calculate the latent space for given images set
def get_Latent_Space(models,images):
  z_means = models['z_meaner'].predict(images)
  z_log_vars=models['z_log_varer'].predict(images)
  std_of_mu=z_means.std(axis=0)
  mean_of_var=(np.exp(z_log_vars/2)).mean(axis=0)
  Latent_SNR=std_of_mu/mean_of_var
  return z_means,z_log_vars,Latent_SNR

def show_train_stats(models,Latent_Space,logs_filename,images,start_epoch=0,stop_epoch=1000,Latent_SNR_sorted=True,Loss_type='Original',Beta=10**-2,free_bits=40,image_size=64,batch_size=32):
  z_means,z_log_vars,Latent_SNR=Latent_Space
  fig,ax=plt.subplots(1,4,figsize=(30,5))

  #Plot learning curve
  logs_file=open(logs_filename)
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
  ax[0].plot(10*np.arange(start_index,stop_index),loss[start_index:stop_index],label='Train')
  ax[0].plot(10*np.arange(start_index,stop_index),val_loss[start_index:stop_index],label='Validation')
  ax[0].set_ylabel('Loss')
  ax[0].set_xlabel('epoch number')
  ax[0].set_title('Learning curve')
  ax[0].legend()

  if Latent_SNR_sorted:
    g=sns.barplot(ax=ax[1],x=np.linspace(0,64,64),y=np.sort(Latent_SNR)[::-1])
  else:
    g=sns.barplot(ax=ax[1],x=np.linspace(0,64,64),y=Latent_SNR)
  ax[1].hlines(1,0,63,label=r'$\mu_{std}=\sigma_{mean}$')
  ax[1].legend()
  ax[1].set_xticks([])
  ax[1].set_xlabel('Latent variable')
  ax[1].set_ylabel('Ratio')
  ax[1].set_title(r'Latent SNR = $\mu_{std}/\sigma_{mean}$')

  #Plot Binary_CE and KL_divergence histograms

  #Reconstruction quality
  flattened_x=K.reshape(images,shape=(len(images),image_size*image_size))
  flattened_decoded=K.reshape(models['vae'].predict(images),shape=(len(images),image_size*image_size))
  Log_loss=image_size*image_size*tf.keras.losses.binary_crossentropy(flattened_x,flattened_decoded)

  #Latent space regularization
  KL_loss=0.5 * K.sum(1 + z_log_vars - K.square(z_means) - K.exp(z_log_vars), axis=-1)

  ax[2].hist(Log_loss/image_size/image_size,bins=100)
  ax[2].set_title('Binary crossentropy')
  ax[2].set_xlabel('Loss per image pixel')

  ax[3].hist(KL_loss/image_size/image_size,bins=100)
  ax[3].set_title('KL divergence')
  ax[3].set_xlabel('Loss per image pixel')
  if Loss_type=='Beta':
    print('Loss uses {:.2f}xKL_divergence'.format(Beta))
  elif Loss_type=='Flow':
    print('Loss uses max(-{:.2f},KL_divergence)'.format(free_bits))

  plt.show()

def rotational_invariance_and_denoising(models,images):
  imgs_turned=copy.deepcopy(images)
  imgs_turned[0]=images[0,::-1,:,:]
  imgs_turned[1]=images[0]
  imgs_turned[2:5]=[images[0]+np.random.normal(0,1/i,images[0].shape) for i in range(60,10,-20)]
  decoded_turned= models['vae'].predict(imgs_turned)
  plot_galaxies(imgs_turned[:5],decoded_turned[:5])


def galaxy_properties_from_latent_variables(models,Latent_Space,number_of_z_to_consider=64):
  z_means,z_log_vars,Latent_SNR=Latent_Space
  
  latent_average=z_means.mean(axis=0)
  variables_to_consider=np.argsort(Latent_SNR)[::-1][:number_of_z_to_consider]
  variances_to_consider=np.exp(z_log_vars.std(axis=0)[variables_to_consider]/2)
  z_to_consider=np.zeros((number_of_z_to_consider,7,64))
  #i - number of variable to change
  for i in range(number_of_z_to_consider):
    #j - number of sigmas to add to it
    for j in range(7):
      #Assign everything to be like average galaxy
      z_to_consider[i,j,:]=latent_average
      #Vary one of the variables to get mu+-(0,1,2,3)*sigma
      z_to_consider[i,j,variables_to_consider[i]]+=(j-3)*variances_to_consider[i]
  images_to_consider=models['decoder'].predict(z_to_consider.reshape((number_of_z_to_consider*7,64)))
  for i in range(number_of_z_to_consider):
    plot_galaxies(images_to_consider[i*7:(i+1)*7])

def latent_distribution(Latent_Space,variable='mean'):
  z_means,z_log_vars,Latent_SNR=Latent_Space
  fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(20, 20))
  axs_flat = axs.flatten()  
  for i in range(z_means.shape[1]):
    if variable=='mean':
      axs_flat[i].hist(z_means[:,i], bins=50)
    else:
      axs_flat[i].hist(z_log_vars[:,i], bins=50)
    axs_flat[i].set_title('Z {}'.format(i))
    axs_flat[i].get_yaxis().set_visible(False)
  fig.tight_layout()
