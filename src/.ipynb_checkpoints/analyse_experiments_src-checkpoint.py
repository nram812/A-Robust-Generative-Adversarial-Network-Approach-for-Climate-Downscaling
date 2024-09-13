import xarray as xr
import sys
import matplotlib.pyplot as plt
import os
import tqdm
from tqdm import tqdm
import tqdm
import tensorflow as tf
import numpy as np
#DisableqdGPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from comet_ml import Experiment
import numpy as np
import tensorflow as tf
import albumentations as A
from functools import partial
AUTOTUNE = tf.data.experimental.AUTOTUNE
from dask.diagnostics import ProgressBar
import json
import pandas as pd
# Create an experiment with your api key
sys.path.append(r'/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN')
from src_unet_init_step import *
from tensorflow.keras import layers
import pandas as pd
# changed activation function to hyperbolic tangent
def load_model_cascade_tanh(model_name, epoch, model_dir):

    gan = tf.keras.models.load_model(f'{model_dir}/{model_name}/generator_epoch_{epoch}.h5', custom_objects={"BicubicUpSampling2D":BicubicUpSampling2D,
                                                                                                                                    "<lambda>": lambda x: 2.0* tf.keras.activations.tanh(x/6.5)}, compile =False)
    unet = tf.keras.models.load_model(f'{model_dir}/{model_name}/unet_epoch_{epoch}.h5', custom_objects={"BicubicUpSampling2D":BicubicUpSampling2D,
                                                                                                                                    "<lambda>": lambda x: 8.25* tf.keras.activations.tanh(x/2.5)}, compile =False)
    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)
    return gan, unet, config["ad_loss_factor"]


def load_model_cascade(model_name, epoch, model_dir, load_unet =True):

    gan = tf.keras.models.load_model(f'{model_dir}/{model_name}/generator_epoch_{epoch}.h5', custom_objects={"BicubicUpSampling2D":BicubicUpSampling2D,
                                                                                                                              }, compile =False)
    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)
    if load_unet:
        unet = tf.keras.models.load_model(f'{model_dir}/{model_name}/unet_epoch_{epoch}.h5',
                                          custom_objects={"BicubicUpSampling2D": BicubicUpSampling2D}, compile=False)

        return gan, unet, config["ad_loss_factor"]
    else:
        return gan, config["ad_loss_factor"]


def load_model_reg(model_name, epoch, model_dir):

    gan = tf.keras.models.load_model(f'{model_dir}/{model_name}/generator_epoch_{epoch}.h5', custom_objects={"BicubicUpSampling2D":BicubicUpSampling2D,
                                                                                                                                    "<lambda>": lambda x: 8.25* tf.keras.activations.tanh(x/6.5)}, compile =False)

    with open(f'{model_dir}/{model_name}/config_info.json') as f:
        config = json.load(f)
    return gan, config["ad_loss_factor"]


def load_trainig_data_historical(model):
    # Define the base paths
    raw_gcm_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_GCM_fields/'
    processed_data_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN/inputs/'
    ccam_output_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_CCAM_outputs/'

    # Load and process raw precipitation fields
    #df_raw_ssp370 = xr.open_dataset(f'{raw_gcm_path}{model}_ssp370_raw_GCM_pr.nc').sel(lat=slice(-65, -25), lon=slice(150, 220.5))
    df_raw_historical = xr.open_dataset(f'{raw_gcm_path}{model}_historical_raw_GCM_pr.nc').sel(lat=slice(-65, -25), lon=slice(150, 220.5))
    
    #df_raw_ssp370['time'] = pd.to_datetime(df_raw_ssp370.time.dt.strftime('%Y-%m-%d'))
    df_raw_historical['time'] = pd.to_datetime(df_raw_historical.time.dt.strftime('%Y-%m-%d'))
    
    df_raw =df_raw_historical# #xr.concat([df_raw_historical.sel(time=slice(None, "2014")),
                        #df_raw_ssp370.sel(time=slice("2015", None))], dim="time")
                    

    # Load future prediction fields
    with ProgressBar():
        df = xr.open_dataset(f'{processed_data_path}/{model}_histupdated.nc', chunks = {"time":365*10}).load()
        df = df.resample(time ='1D').mean()
    #df_future = xr.open_dataset(f'{processed_data_path}/{model}_ssp370updated.nc').resample(time ='1D').mean()
    # df = xr.concat([df.sel(time=slice(None, "2014")),
    #                     df_future.sel(time=slice("2015", None))], dim="time")
    df['time'] = pd.to_datetime(df.time.dt.strftime("%Y-%m-%d"))
    #df = df.resample(time ='1D').mean()

    # Load and process CCAM outputs
    #y_ssp370 = xr.open_dataset(f'{ccam_output_path}{model}_ssp370_precip.nc')
    y_historical = xr.open_dataset(f'{ccam_output_path}{model}_historical_precip.nc')
    
    #y_ssp370['time'] = pd.to_datetime(y_ssp370.time.dt.strftime("%Y-%m-%d"))
    y_historical['time'] = pd.to_datetime(y_historical.time.dt.strftime("%Y-%m-%d"))
    
    y = y_historical#xr.concat([y_historical.sel(time=slice(None, "2014")), y_ssp370.sel(time=slice("2015", None))], dim="time")

    return df_raw, df, y


def load_and_concatenate_hist(model):
    # Define the base paths
    raw_gcm_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_GCM_fields/'
    processed_data_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/Processed_CMIP6_DATA/'
    ccam_output_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_CCAM_outputs/'

    # Load and process raw precipitation fields
    #df_raw_ssp370 = xr.open_dataset(f'{raw_gcm_path}{model}_ssp370_raw_GCM_pr.nc').sel(lat=slice(-65, -25), lon=slice(150, 220.5))
    df_raw_historical = xr.open_dataset(f'{raw_gcm_path}{model}_historical_raw_GCM_pr.nc').sel(lat=slice(-65, -25), lon=slice(150, 220.5))
    
    #df_raw_ssp370['time'] = pd.to_datetime(df_raw_ssp370.time.dt.strftime('%Y-%m-%d'))
    df_raw_historical['time'] = pd.to_datetime(df_raw_historical.time.dt.strftime('%Y-%m-%d'))
    
    df_raw = df_raw_historical#xr.concat([df_raw_historical.sel(time=slice(None, "2014")),
                        #df_raw_ssp370.sel(time=slice("2015", None))], dim="time")

    # Load future prediction fields
    #df_ssp370 = xr.open_dataset(f'{processed_data_path}Processed_{model}_ssp370.nc')
    df_historical = xr.open_dataset(f'{processed_data_path}Processed_{model}_historical.nc')
    
    df = df_historical#xr.concat([df_historical.sel(time=slice(None, "2014")), df_ssp370.sel(time=slice("2015", None))], dim="time")

    # Load and process CCAM outputs
    #y_ssp370 = xr.open_dataset(f'{ccam_output_path}{model}_ssp370_precip.nc')
    y_historical = xr.open_dataset(f'{ccam_output_path}{model}_historical_precip.nc')
    
    #y_ssp370['time'] = pd.to_datetime(y_ssp370.time.dt.strftime("%Y-%m-%d"))
    y_historical['time'] = pd.to_datetime(y_historical.time.dt.strftime("%Y-%m-%d"))
    
    y = y_historical#xr.concat([y_historical.sel(time=slice(None, "2014")), y_ssp370.sel(time=slice("2015", None))], dim="time")

    return df_raw, df, y


def load_and_concatenate_datasets(model):
    # Define the base paths
    raw_gcm_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_GCM_fields/'
    processed_data_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/inputs/Processed_CMIP6_DATA/'
    ccam_output_path = '/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_CCAM_outputs/'

    # Load and process raw precipitation fields
    df_raw_ssp370 = xr.open_dataset(f'{raw_gcm_path}{model}_ssp370_raw_GCM_pr.nc').sel(lat=slice(-65, -25), lon=slice(150, 220.5))
    df_raw_historical = xr.open_dataset(f'{raw_gcm_path}{model}_historical_raw_GCM_pr.nc').sel(lat=slice(-65, -25), lon=slice(150, 220.5))
    
    df_raw_ssp370['time'] = pd.to_datetime(df_raw_ssp370.time.dt.strftime('%Y-%m-%d'))
    df_raw_historical['time'] = pd.to_datetime(df_raw_historical.time.dt.strftime('%Y-%m-%d'))
    
    df_raw = xr.concat([df_raw_historical.sel(time=slice(None, "2014")),
                        df_raw_ssp370.sel(time=slice("2015", None))], dim="time")

    # Load future prediction fields
    df_ssp370 = xr.open_dataset(f'{processed_data_path}Processed_{model}_ssp370.nc')
    df_historical = xr.open_dataset(f'{processed_data_path}Processed_{model}_historical.nc')
    
    df = xr.concat([df_historical.sel(time=slice(None, "2014")), df_ssp370.sel(time=slice("2015", None))], dim="time")

    # Load and process CCAM outputs
    y_ssp370 = xr.open_dataset(f'{ccam_output_path}{model}_ssp370_precip.nc')
    y_historical = xr.open_dataset(f'{ccam_output_path}{model}_historical_precip.nc')
    
    y_ssp370['time'] = pd.to_datetime(y_ssp370.time.dt.strftime("%Y-%m-%d"))
    y_historical['time'] = pd.to_datetime(y_historical.time.dt.strftime("%Y-%m-%d"))
    
    y = xr.concat([y_historical.sel(time=slice(None, "2014")), y_ssp370.sel(time=slice("2015", None))], dim="time")

    return df_raw, df, y

def load_and_normalize_topography_data(filepath):
    # Load the dataset
    topography_data = xr.open_dataset(filepath)

    # Extract variables
    vegt = topography_data.vegt
    orog = topography_data.orog
    he = topography_data.he

    # Print maximum values
    print(f"Max orog: {orog.max().values}, Max he: {he.max().values}, Max vegt: {vegt.max().values}")

    # Normalize the data to the range [0, 1]
    vegt = (vegt - vegt.min()) / (vegt.max() - vegt.min())
    orog = (orog - orog.min()) / (orog.max() - orog.min())
    he = (he - he.min()) / (he.max() - he.min())

    return vegt, orog, he
import pandas as pd

def normalize_and_stack(concat_dataset, means_filepath, stds_filepath, variables):
    """
    Normalizes specified variables in a dataset with given mean and standard deviation,
    then stacks them along a new 'channel' dimension.

    Parameters:
    concat_dataset (xarray.Dataset): Dataset to normalize.
    means_filepath (str): File path to the dataset containing mean values.
    stds_filepath (str): File path to the dataset containing standard deviation values.
    variables (list): List of variable names to normalize and stack.

    Returns:
    xarray.Dataset: The normalized and stacked dataset.
    """

    # Load mean and standard deviation datasets
    means = xr.open_dataset(means_filepath)
    stds = xr.open_dataset(stds_filepath)

    # Normalize the dataset
    X_norm = (concat_dataset[variables] - means) / stds
    X_norm['time'] = pd.to_datetime(X_norm.time.dt.strftime("%Y-%m-%d"))

    # Stack the variables along a new 'channel' dimension
    stacked_X = xr.concat([X_norm[varname] for varname in variables], dim="channel")
    stacked_X['channel'] = (('channel'), variables)
    stacked_X = stacked_X.transpose("time", "lat", "lon", "channel")

    return stacked_X


@tf.function
def predict_batch(model, latent_vectors, data_batch, orog, he, vegt, model_type):
    if model_type =='GAN':
        return model([latent_vectors[0],  data_batch, orog, he, vegt], training=False)
    else:
        return model(data_batch, training=False)
    
def expand_conditional_inputs(X, batch_size):
    expanded_image = tf.expand_dims(X, axis=0)  # Shape: (1, 172, 179)

# Repeat the image to match the desired batch size
    expanded_image = tf.repeat(expanded_image, repeats=batch_size, axis=0)  # Shape: (batch_size, 172, 179)

    # Create a new axis (1) on the last axis
    expanded_image = tf.expand_dims(expanded_image, axis=-1) 
    return expanded_image


def predict_parallel_v1(model, inputs, output_shape, batch_size,orog_vector, he_vector, vegt_vector, model_type ='GAN'):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    dset = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            if model_type =='GAN':
                random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
                #random_latent_vectors2 = tf.random.normal(shape=(1,) + tuple(model.inputs[1].shape[1:]))
                random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
                #random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
                orog = expand_conditional_inputs(orog_vector, batch_size)
                he =expand_conditional_inputs(he_vector, batch_size) 
                vegt = expand_conditional_inputs(vegt_vector, batch_size)#ex, he_vector, vegt_vector
                
            else:
                random_latent_vectors1 =[]
                random_latent_vectors2 =[]

            output = predict_batch(model, [random_latent_vectors1], data_batch, orog, he, vegt, model_type)

            dset+=(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()
            pbar.update(1)  # Update the progress bar

    if remainder != 0:
        if model_type =='GAN':
            random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
            #random_latent_vectors2 = tf.random.normal(shape=(1,) + tuple(model.inputs[1].shape[1:]))
            random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            #random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, remainder)
            he =expand_conditional_inputs(he_vector, remainder) 
            vegt = expand_conditional_inputs(vegt_vector, remainder)
        else:
            random_latent_vectors1 =[]
            random_latent_vectors2 =[]
        output = predict_batch(model, [random_latent_vectors1[:remainder]],
                               inputs[inputs.shape[0] - remainder:],orog, he, vegt, model_type)
       
        dset+=(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()

    #dset = np.array(dset)

    # Assuming 'y' is defined somewhere
    output_shape['pr'].values = dset

    return output_shape
# Example usage:
@tf.function
def predict_batch_residual(model, unet, latent_vectors, data_batch, orog, he, vegt, model_type):
    if model_type =='GAN':
        intermediate =unet([latent_vectors[0],  data_batch, orog, he, vegt], training=False)
        #intermediate = apply_gaussian_blur(intermediate, size=7, sigma=1.5)
        #max_value = tf.reduce_max(intermediate, axis=(1, 2, 3), keepdims=True)
        #min_value = tf.reduce_min(intermediate, axis=(1, 2, 3), keepdims=True)
        init_prediction = intermediate#(intermediate - min_value)/(max_value - min_value)
        #print(intermediate)
        #intermediate = tf.cast(tf.math.sqrt(tf.clip_by_value(intermediate, clip_value_min=0, clip_value_max=2500)), 'float32')
        return  model([latent_vectors[0],  data_batch, orog, he, vegt, init_prediction], training=False) +intermediate#+
    else:
        return unet([latent_vectors[0],  data_batch, orog, he, vegt], training=False)
@tf.function
def predict_batch22(model, unet, latent_vectors, data_batch, orog, he, vegt, model_type):
    if model_type =='GAN':
        intermediate =unet([latent_vectors[0],  data_batch, orog, he, vegt], training=False)
        #print(intermediate)
        #intermediate = tf.cast(tf.math.sqrt(tf.clip_by_value(intermediate, clip_value_min=0, clip_value_max=2500)), 'float32')
        return model([latent_vectors[0],  data_batch, orog, he, vegt, intermediate], training=False)
    else:
        return model(data_batch, training=False)
def predict_parallel_v2(model, unet, inputs, output_shape, batch_size,orog_vector, he_vector, vegt_vector, model_type ='GAN'):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    dset = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
            if model_type =='GAN':
                random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
                #random_latent_vectors2 = tf.random.normal(shape=(1,) + tuple(model.inputs[1].shape[1:]))
                random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
                #random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
                orog = expand_conditional_inputs(orog_vector, batch_size)
                he =expand_conditional_inputs(he_vector, batch_size) 
                vegt = expand_conditional_inputs(vegt_vector, batch_size)#ex, he_vector, vegt_vector
                
            else:
                random_latent_vectors1 =[]
                random_latent_vectors2 =[]

            output = predict_batch22(model,unet,  [random_latent_vectors1], data_batch, orog, he, vegt, model_type)

            dset+=(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()#output.numpy()[:,:,:,0].tolist()##output.numpy()[:,:,:,0].tolist()
            pbar.update(1)  # Update the progress bar

    if remainder != 0:
        if model_type =='GAN':
            random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
            #random_latent_vectors2 = tf.random.normal(shape=(1,) + tuple(model.inputs[1].shape[1:]))
            random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            #random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, remainder)
            he =expand_conditional_inputs(he_vector, remainder) 
            vegt = expand_conditional_inputs(vegt_vector, remainder)
        else:
            random_latent_vectors1 =[]
            random_latent_vectors2 =[]
        output = predict_batch22(model, unet, [random_latent_vectors1[:remainder]],
                               inputs[inputs.shape[0] - remainder:],orog, he, vegt, model_type)
       
        dset+=(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()#output.numpy()[:,:,:,0].tolist()#(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()

    #dset = np.array(dset)

    # Assuming 'y' is defined somewhere
    output_shape['pr'].values = dset

    return output_shape



def predict_parallel_resid(model, unet, inputs, output_shape, batch_size,orog_vector, he_vector, vegt_vector, model_type ='GAN'):
    n_iterations = inputs.shape[0] // batch_size
    remainder = inputs.shape[0] - n_iterations * batch_size

    dset = []

    with tqdm.tqdm(total=n_iterations, desc="Predicting", unit="batch") as pbar:
        for i in range(n_iterations):
            data_batch = inputs[i * batch_size: (i + 1) * batch_size]
     
            random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
            #random_latent_vectors2 = tf.random.normal(shape=(1,) + tuple(model.inputs[1].shape[1:]))
            random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
            #random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
            orog = expand_conditional_inputs(orog_vector, batch_size)
            he =expand_conditional_inputs(he_vector, batch_size) 
            vegt = expand_conditional_inputs(vegt_vector, batch_size)#ex, he_vector, vegt_vector

            output = predict_batch_residual(model,unet,  [random_latent_vectors1], data_batch, orog, he, vegt, model_type)

            dset+=(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()#output.numpy()[:,:,:,0].tolist()##output.numpy()[:,:,:,0].tolist()
            pbar.update(1)  # Update the progress bar

    if remainder != 0:
       
        random_latent_vectors1 = tf.random.normal(shape=(1,) + tuple(model.inputs[0].shape[1:]))
        #random_latent_vectors2 = tf.random.normal(shape=(1,) + tuple(model.inputs[1].shape[1:]))
        random_latent_vectors1 = tf.repeat(random_latent_vectors1, repeats=batch_size, axis=0)
        #random_latent_vectors2 = tf.repeat(random_latent_vectors2, repeats=batch_size, axis=0)
        orog = expand_conditional_inputs(orog_vector, remainder)
        he =expand_conditional_inputs(he_vector, remainder) 
        vegt = expand_conditional_inputs(vegt_vector, remainder)

        output = predict_batch_residual(model, unet, [random_latent_vectors1[:remainder]],
                               inputs[inputs.shape[0] - remainder:],orog, he, vegt, model_type)
       
        dset+=(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()#output.numpy()[:,:,:,0].tolist()#(np.exp(output.numpy()[:,:,:,0])-0.001).tolist()

    #dset = np.array(dset)

    # Assuming 'y' is defined somewhere
    output_shape['pr'].values = dset

    return output_shape



import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar



def preprocess(dict_path, observed_path=r"/nesi/project/niwa00004/ML_DATA/VCSN/Augmented/vcsn_rainfall_augmented.nc"):
    array_list = []
    for key_name, path in dict_path.items():
        print(key_name)
        predicted = xr.open_dataset(path, chunks ={"time":1000})
        try:
            predicted = predicted.drop(["lat","lon"]).rename({"pr":"rain_council","latitude":"lat", "longitude":"lon"})
        except:
            predicted = predicted#.rename({"latitude":"lat", "longitude":"lon"})
        try:
            predicted = predicted.rename({"latitude":"lat", "longitude":"lon"})
        except:
            predicted = predicted.rename({"pr":"rain_council"})
        predicted = predicted.reindex(lon = sorted(predicted.lon.values))
        predicted = predicted.expand_dims({"experiment":1})
        predicted['experiment'] = (('experiment'), [key_name])
        # append the array list to the paht
        array_list.append(predicted)
    concat_array = xr.concat(array_list, dim ="experiment")
    
    # load observed data
    observed_data = xr.open_dataset(observed_path, chunks ={"time":1000}).rename({"latitude":"lat", "longitude":"lon"})
    observed_data= observed_data.reindex(lon = sorted(observed_data.lon.values))
    # time for the VCSN is in UTC time
    observed_data['time'] = pd.to_datetime(observed_data.time.dt.strftime("%Y-%m-%d"))
    observed_data = observed_data.sel(time = concat_array.time)
    # modifying the fill_value criterion
    concat_array = concat_array#.interp_like(observed_data.rain_council.isel(time =0),
                                #            method ='nearest', kwargs = dict(fill_value = 'extrapolate'))
    return concat_array, observed_data
        

def load_comparison_netcdfs(model, ssp, time_slice =slice("1975","2014"), model_name =None, base_dirs = None):
    # multiplication by GCM outputs as the precipitation flux doesn't have the currect units
    raw_gcm = format_daily_time(xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_GCM_fields/{model}_{ssp}_raw_GCM_pr.nc',
                                                chunks={"time": 5000})).sel(lat=slice(-65, -25), lon=slice(160, 190))
    # converting the timestep so that it is consistent
    ccam_outputs = format_daily_time(xr.open_dataset(f'/nesi/project/niwa00018/ML_downscaling_CCAM/Training_CNN/outputs/raw_CCAM_outputs/{model}_{ssp}_precip.nc',
                                                     chunks={"time": 5000}))
    # loading the observatiosn
    observations = format_daily_time(
        xr.open_dataset(r"/nesi/project/niwa00004/ML_DATA/VCSN/Augmented/vcsn_rainfall_augmented.nc",
                        chunks={"time": 5000})).sel(time = time_slice)
    # ml_outputs
    ml_outputs = format_daily_time(
        xr.open_dataset(f'{base_dirs}/ml_downscaled_outputs/{model_name}/{model}_{ssp}_ml_downscaled_rainfall_updated_{model_name}.nc',
                        chunks={"time": 5000}))
    if ssp =="historical":
    # print(ml_outputs, observations, ccam_outputs,
        common_time_with_obs = ml_outputs.time.to_index() \
            .intersection(ccam_outputs.time.to_index()) \
            .intersection(raw_gcm.time.to_index()) \
            .intersection(observations.time.to_index())
    else:
        common_time = ml_outputs.time.to_index() \
            .intersection(ccam_outputs.time.to_index()) \
            .intersection(raw_gcm.time.to_index())
        common_time_with_obs = ml_outputs.time.to_index() \
            .intersection(ccam_outputs.time.to_index()) \
            .intersection(raw_gcm.time.to_index()) \
            .intersection(observations.time.to_index())
    # identifying the common times across all the indices
    # renaming some of the variable names also
    if ssp =="historical":
        with ProgressBar():
            observations = observations.sel(time=common_time_with_obs)[["rain_council"]].compute().rename({"latitude":"lat","longitude":"lon"})
            ml_outputs = ml_outputs.sel(time=common_time_with_obs).rename({model_name: "rain_council"})[
                ["rain_council"]].compute().rename({"latitude":"lat","longitude":"lon"})
            raw_gcm = raw_gcm.sel(time=common_time_with_obs).rename({"pr": "rain_council"})[["rain_council"]].compute()  * 3600*24
            ccam_outputs = ccam_outputs.sel(time=common_time_with_obs).rename({"pr": "rain_council"})[["rain_council"]].compute()
    else:
        # we don't save the observations as an output
        with ProgressBar():
            observations = observations.sel(time=common_time_with_obs)[["rain_council"]].compute().rename(
                {"latitude": "lat", "longitude": "lon"})

            ml_outputs = ml_outputs.sel(time=common_time).rename({model_name: "rain_council"})[
                ["rain_council"]].compute().rename({"latitude":"lat","longitude":"lon"})
            raw_gcm = raw_gcm.sel(time=common_time).rename({"pr": "rain_council"})[["rain_council"]].compute()  * 3600*24
            ccam_outputs = ccam_outputs.sel(time=common_time).rename({"pr": "rain_council"})[["rain_council"]].compute()


    return {"ml_output": ml_outputs, "raw_gcm": raw_gcm, "ccam": ccam_outputs, "obs": observations}
    # subsetting the data for all common_times



class ValidationMetric(object):
    """
    This is a class that computes a wide variety of different metrics for validating a series
    """
    def __init__(self, datasets):
        self.ds = datasets

    def __call__(self, thresh):
        print("Computing Indices.....annual_rainfall")
        annual_rainfall = self.seasonal_rainfall(self.ds)
        print("Computing Indices.....CDD")
        cdd = self.consecutive_dry_days(self.ds, thresh)
        print("Computing Indices.....RX3DAY")
        rx3day = self.rx1day(self.ds, thresh)

        print("Computing R10 Day.....")
        r10day = self.r10day(self.ds)
        merged_df = xr.merge([cdd, rx3day,annual_rainfall,r10day])
        return merged_df

    @staticmethod
    def consecutive_dry_days(ds, thresh=1):
        """
        Compute the number of consecutive dry days in a year for a gridded dataset

        Parameters:
        ds (xarray.Dataset): Gridded dataset with a time dimension
        thresh (float): Threshold value for defining a dry day (default: 0.1)
        time_dim (str): Name of the time dimension in the dataset (default: 'time')

        Returns:
        xarray.DataArray: Number of consecutive dry days in a year
        """

        # Create a function to find consecutive True values in a boolean array
        def find_consecutive_true(arr):
            if ((arr.max() ==1) & (arr.min()==0))|(arr.min()==1):
                arr = np.asarray(arr)
                idx = np.flatnonzero(np.concatenate(([arr[0]],
                                                     arr[:-1] != arr[1:],
                                                     [True])))

                z = np.diff(idx)[::2]
                return np.max(z, axis=0)
            else:
                # this condition implies that there are no CDD throughout a year?
                return 0.0

        test_data = ds.pr
        try:
            test_data = test_data.stack(z=['lat','lon']).dropna("z")
        except:
            test_data = test_data.stack(z=['latitude', 'longitude']).dropna("z")
        #fillna(-999)
        bool_arr = (test_data <= thresh).astype('int')
        bool_arr = bool_arr
        with ProgressBar():
            consec_dry_days = xr.apply_ufunc(find_consecutive_true, bool_arr.groupby('time.year'),                                             input_core_dims=[["time"]], output_core_dims=[[]],
                                             output_dtypes=[int], vectorize=True, dask='parallelized').compute()
        consec_dry_days = consec_dry_days.unstack()
        try:
            consec_dry_days = consec_dry_days.reindex(lat=sorted(consec_dry_days.lat.values))

        except:
            consec_dry_days = consec_dry_days.reindex(longitude=sorted(consec_dry_days.longitude.values))

        output = consec_dry_days

        return output.to_dataset().rename({"pr":"cdd"})

    @staticmethod
    def rx1day(ds,thresh=1):
        """
        Compute the Rx3day index for a gridded dataset

        Parameters:
        ds (xarray.Dataset): Gridded dataset with a time dimension
        thresh (float): Threshold value for defining a wet day (default: 0.1)
        time_dim (str): Name of the time dimension in the dataset (default: 'time')

        Returns:
        xarray.DataArray: Rx3day index for the dataset
        """
        return ds.groupby('time.year').max().rename({"pr":"rx1day"})
    
    @staticmethod
    def seasonal_rainfall(ds):
        output = ds.groupby('time.season').mean()#.mean("year")
        output1 = output.sel(season = 'DJF').drop("season").rename({"pr":"DJF_rainfall"})
        output2 = output.sel(season = 'JJA').drop("season").rename({"pr":"JJA_rainfall"})
        output = xr.merge([output1, output2])

        return output
    @staticmethod
    def r10day(ds):
        output = (ds>10).groupby('time.year').sum()#.mean("year")
        return output.rename({"pr":"r10day"})

#     @staticmethod
#     def sdii(ds, output_grid, thresh=1):
#         output = ds.where(ds > thresh, np.nan).mean("time")
#         output = output.ffill("lat", limit=5)\
#             .ffill("lon", limit=5)\
#             .interp(lat=output_grid.lat,
#                     lon=output_grid.lon,
#                     method='nearest',
#                     kwargs=dict(fill_value='extrapolate'))
#         output = output.where(~output_grid.rain_council.isel(time=0).isnull(), np.nan)
#         return output.rename({"rain_council":"sdii"})

#     @staticmethod
#     def pd1mm(ds, output_grid, thresh =1):
#         output = (ds > thresh).groupby('time.year').sum().mean("year")
#         output = output.ffill("lat", limit=5)\
#             .ffill("lon", limit=5)\
#             .interp(lat=output_grid.lat,
#                     lon=output_grid.lon,
#                     method='nearest',
#                     kwargs=dict(fill_value='extrapolate'))
#         output = output.where(~output_grid.rain_council.isel(time=0).isnull(), np.nan)

#         return output.rename({"rain_council":"pd1mm"})
    # @staticmethod
    # def rx3day(ds,thresh=1):
    #     """
    #     Compute the Rx3day index for a gridded dataset

    #     Parameters:
    #     ds (xarray.Dataset): Gridded dataset with a time dimension
    #     thresh (float): Threshold value for defining a wet day (default: 0.1)
    #     time_dim (str): Name of the time dimension in the dataset (default: 'time')

    #     Returns:
    #     xarray.DataArray: Rx3day index for the dataset
    #     """

    #     rolling_sum = ds.rolling({"time": 3}, min_periods=3).sum()
    #     rolling_binary_mask = (ds > thresh).rolling({"time": 3}, min_periods=3).sum()

    #     rolling_sum = rolling_sum.where(rolling_binary_mask == 3, np.nan)
    #     annual_maximum = rolling_sum.groupby('time.year').max()#.mean("year")

    #     annual_maximum = annual_maximum

    #     return annual_maximum.rename({"pr":"rx3day"})


                                                 
def run_experiments(experiments, epoch_list, model_dir, 
                    input_predictors, common_times, output_shape, orog, he, vegt, n_members, batch_size =64):
    """
    Runs inference on some predictor fields in/ out-of-sample


    experiments: list of experiment names in the model_dir folder
    input_predictors: stacked netcdf with dims (time, lat, lon, channel) and normalized data
    common_times: common_times between output_shape data and input_predictors
    output_shape: a netcdf (y_true) that is the same shape as the output prediction, it contains the time metadata
    orog, he, vegt: auxiliary files from ccam
    n_member: the number of ensemble members

    """
    
    # if the epoch list is only a float convert to a list
    if isinstance(epoch_list, int):
        epoch_list = [epoch_list] * len(experiments)

    # creating empty lists to save outputs            
    dsets = []
    lambda_var =[]
    for i, experiment in enumerate(experiments):
        if 'cascade' in experiment:
            gan, unet, lambdas = load_model_cascade(experiment,
                                                epoch_list[i], model_dir)
            if i ==0:
                # first instance is always a unet model
                lambdas =0.0
                preds = xr.concat([predict_parallel_resid(gan,unet,
                                                        input_predictors.sel(time = common_times).values,
                                                        output_shape.sel(time = common_times),
                                                        batch_size,orog, he, vegt,model_type ='unet') 
                                                        for i in range(n_members)],
                                                          dim ="member")
            else:
                # do not change lambdas value otherwise
                lambdas = lambdas
                preds = xr.concat([predict_parallel_resid(gan,unet,
                                                        input_predictors.sel(time = common_times).values,
                                                        output_shape.sel(time = common_times),
                                                        batch_size,orog, he, vegt,model_type ='GAN') 
                                                        for i in range(n_members)],
                                                          dim ="member")


        else:
            gan, lambdas = load_model_reg(experiment,
                                                epoch_list[i], model_dir)
            preds = xr.concat([predict_parallel_v1(gan,
                                        input_predictors.sel(time = common_times).values,
                                        output_shape.sel(time = common_times),
                                        batch_size,orog, he, vegt,model_type ='GAN') 
                                        for i in range(n_members)],
                                            dim ="member")
        lambda_var.append(lambdas)
        dsets.append(preds)
            # finish the experiment and concatenate the data
    dsets = xr.concat(dsets, dim ="experiment")
    dsets['experiment'] = (('experiment'), lambda_var)
    dsets = dsets.reindex(experiment = sorted(dsets.experiment.values)) 
    return dsets
