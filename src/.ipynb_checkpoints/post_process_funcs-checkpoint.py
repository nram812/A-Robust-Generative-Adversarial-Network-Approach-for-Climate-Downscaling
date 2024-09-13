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
#RMSE2 = abs(concat_dsets2 - concat_dsets2.sel(model ='Ground Truth')).mean(["lat","lon"])
import cartopy.crs as ccrs
import pandas as pd
# Create an experiment with your api key
sys.path.append(r'/nesi/project/niwa00018/ML_downscaling_CCAM/training_GAN')
from src_unet_init_step import *
from paper_experiments.analyse_experiments_src import *
from tensorflow.keras import layers
import pandas as pd
import json


def plot_metric(metrics, ens_mean = False, metric_name ='cdd', cmap ='BrBG',
                    vmin =0, vmax =200, spacing=5, savefig =True, model = None, framework = None, generic_experiment_name = None, output_folder = None):

    if ens_mean:
        metrics = metrics.mean(["year"]).mean("member")
    else:
        metrics = metrics.mean(["year"])    
    

    RMSE_rx1_single = abs(metrics.to_array().sel(variable =metric_name) - metrics.to_array().sel(variable =metric_name).sel(experiment='Ground Truth')).mean(["lat","lon"])
#     pcor_signle= xr.corr(metrics[metric_name].stack(z =['lat','lon']),
#                 metrics.sel(experiment='Ground Truth')[metric_name].stack(z =['lat','lon']), dim ="z")
    RMSE_rx1_single_bias = (metrics.to_array().sel(variable =metric_name) - metrics.to_array().sel(variable =metric_name).sel(experiment='Ground Truth')).mean(["lat","lon"])



    levels = np.arange(vmin, vmax + spacing, spacing)
    fig, ax = plt.subplots(1, 8, subplot_kw=dict(projection = ccrs.PlateCarree(central_longitude =171.77)), figsize = (15, 3), dpi =350)
    #ax = ax.ravel()

    for i,model_name in enumerate(metrics.experiment.values):

        cs = metrics[metric_name].sel(experiment = model_name).plot.contourf( cmap =cmap, vmin =vmin, vmax =vmax, transform = ccrs.PlateCarree(),
                                                                                ax = ax[i], add_colorbar =False, levels = levels,  extend ='both',antialiased=True)

        if not (i==7):    
            
            ax[i].text(169.5, -55, f"MAE:{'%.2f' % RMSE_rx1_single.sel(experiment = model_name).values}\nMBIAS:{'%.2f' % RMSE_rx1_single_bias.sel(experiment = model_name).values}",
        transform = ccrs.PlateCarree(), color ='k',fontsize =9)
        ax[i].set_title('')
        # if i ==0:
        #     # ax[i].text(150.5, -38, f"{season}", weight='bold',
        #     # transform = ccrs.PlateCarree(), color ='k',  fontsize =15)

        if i ==7:
            ax[i].set_title(model_name)
        else:
            ax[i].set_title('$\lambda_{adv}$=' +f'{model_name}')
        ax[i].coastlines('10m')
    ax4 = fig.add_axes([0.1, 0.07, 0.8, 0.03])



    cbar = fig.colorbar(cs, cax = ax4, orientation ='horizontal') 
    cbar.set_ticks(np.arange(0,220, 20))
    cbar.set_ticklabels([f"%.0f" % s for s in np.arange(0,220, 20)])
    if metric_name == 'rx1day':
        cbar.set_label('RX1Day (mm)', fontsize = 12, weight ='bold')
    elif metric_name =='cdd':
        cbar.set_label('CDD (days)', fontsize = 12, weight ='bold')
    else:
        cbar.set_label('Precipitation (mm/day)', fontsize = 12, weight ='bold')
    if savefig:
        if not os.path.exists(f'{output_folder}/{generic_experiment_name}'):
            os.makedirs(f'{output_folder}/{generic_experiment_name}')
        fig.savefig(f'{output_folder}/{generic_experiment_name}/cascaded_model{metric_name}_{model}_{framework}_ensmean_{ens_mean}.png', dpi =500, bbox_inches ='tight')
        return None
    else:
        return fig
def psd(y, bins=np.arange(0, 0.52, 0.02)):
    """
    Compute Power Spectral Density (PSD) of an image y.

    Args:
    - y: Input image with shape (time, lat, lon).
    - bins: Array of bin edges for binning the wavenumbers.

    Returns:
    - psd_array: Array of PSD values with shape (time, K), where K is sqrt(kx^2 + ky^2),
                representing the wavenumber in X and Y.

    - bin_edges: Bin edges used for binning the wavenumbers.
    """
    # Compute 2D FFT of the input image
    ffts = np.fft.fft2(y)
    ffts = np.fft.fftshift(abs(ffts) ** 2)

    # Compute the frequency grids
    freq = np.fft.fftshift(np.fft.fftfreq(172))
    freq2 = np.fft.fftshift(np.fft.fftfreq(179))
    kx, ky = np.meshgrid(freq, freq2)
    kx = kx.T
    ky = ky.T

    # Compute PSD by binning wavenumbers
    x = [
        binned_statistic(
            np.sqrt(kx.ravel() ** 2 + ky.ravel() ** 2),
            values=np.vstack(ffts[i].ravel()).T,
            statistic="mean",
            bins=bins,
        ).statistic
        for i in range(ffts.shape[0])
    ]

    # Compute PSD for the last time step (for normalization)
    x2 = binned_statistic(
        np.sqrt(kx.ravel() ** 2 + ky.ravel() ** 2),
        values=np.vstack(ffts[-1].ravel()).T,
        statistic="mean",
        bins=bins,
    )

    # Normalize the PSD and return it along with bin edges
    return np.array(x)[:, 0, :] / abs(x2.bin_edges[0] - x2.bin_edges[1]), x2.bin_edges