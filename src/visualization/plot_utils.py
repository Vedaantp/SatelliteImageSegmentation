""" This module contains functions for plotting satellite images. """

import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import xarray as xr

import numpy as np

# local modules
sys.path.append(".")
from src.utilities import get_satellite_dataset_size


def flatten_dataset(data_set: xr.DataArray) -> np.ndarray:
    """
    This function flattens a dataset from its 5D representation to a new
    shape that is tile * time * band * height * width

    Parameters
    ----------
    data_set: xr.DataArray
        The dataset we want to flatten

    Returns
    -------
    np.ndarray
        the new 1D array of shape (tile * time * band * height * width)
    """
    raise NotImplementedError


def flatten_dataset_by_band(data_set: xr.DataArray) -> np.ndarray:
    """
    This function flattens a dataset by band from its 5D representation to a new
    shape that is (band, tile * time  * height * width)

    Parameters
    ----------
    data_set: xr.DataArray
        The dataset we want to flatten

    Returns
    -------
    np.ndarray
        the new array of shape (band, tile * time * height * width)
    """
    raise NotImplementedError


def plot_viirs_histogram(
    data_set: xr.Dataset,
    image_dir: Path = None,
    n_bins=100,
) -> None:
    """
    This function plots the histogram over all VIIRS values.

    Parameters
    ----------
    data_set: xr.Dataset
        The VIIRS satellite dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    num_tiles, num_dates = (
        len(data_set),
        get_satellite_dataset_size(data_set)[0],
    )
    viirs_flattened = flatten_dataset(data_set)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), squeeze=False, tight_layout=True)
    # --- start here ---

    # set scale to log
    
    # plot histogram
    
    # tick params on both axis with size of 20
    
    # set title w/ font size of 12 to: "Viirs Histogram (log scaled), {num_tiles} tiles, {num_dates} dates"
    

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "viirs_histogram.png").touch(exist_ok=True)
        fig.savefig(image_dir / "viirs_histogram.png", format="png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()
    
    raise NotImplementedError


def plot_sentinel1_histogram(
    data_set: xr.Dataset,
    image_dir: Path = None,
    n_bins=20,
) -> None:
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.

    Parameters
    ----------
    data_set: xr.Dataset
        The S1 satellite dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    num_tiles, num_dates, num_bands = (
        len(data_set),
        get_satellite_dataset_size(data_set)[0],
        get_satellite_dataset_size(data_set)[1],
    )
    flattened_dataset_list, satellite_bands = (
        flatten_dataset_by_band(data_set),
        data_set["band"].values,
    )
    fig, ax = plt.subplots(
        num_bands, 1, figsize=(4, 4), squeeze=False, tight_layout=True
    )
    # --- start here ---

    # iterate by band
   
        # set scale to log
        
        # plot histogram
       
        # tick params on both axis with size of 15
        
        # set title w/ font size 9 to: "Sentinel 1 Histogram (log scaled)\n{num_tiles} tiles, {num_dates} dates\nBand {satellite_bands[band]}"
        

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "sentinel1_histogram.png").touch(exist_ok=True)
        fig.savefig(image_dir / "sentinel1_histogram.png", format="png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()

    raise NotImplementedError


def plot_sentinel2_histogram(
    data_set: xr.Dataset,
    image_dir: Path = None,
    n_bins=20,
) -> None:
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    data_set: xr.Dataset
        The S2 satellite dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    num_tiles, num_dates, num_bands = (
        len(data_set),
        get_satellite_dataset_size(data_set)[0],
        get_satellite_dataset_size(data_set)[1],
    )
    flattened_dataset_list, satellite_bands = (
        flatten_dataset_by_band(data_set),
        data_set["band"].values,
    )
    fig, ax = plt.subplots(
        num_bands // 3, 3, figsize=(4,4), squeeze=False
    )
    # --- start here ---

    # iterate by band
   
        # set scale to log
        
        # plot histogram
        
        # tick params on both axis with size of 7
        
        # set title w/ font size 9 to: "Sentinel 2 Histogram (log scaled)\n{num_tiles} tiles, {num_dates} dates\nBand {satellite_bands[band]}"
        

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "sentinel2_histogram.png").touch(exist_ok=True)
        fig.savefig(image_dir / "sentinel2_histogram.png", format="png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()

    raise NotImplementedError


def plot_landsat_histogram(
    data_set: xr.Dataset,
    image_dir: Path = None,
    n_bins=20,
) -> None:
    """
    This function plots the landsat histogram over all Landsat values.

    Parameters
    ----------
    data_set: xr.Dataset
        The Landsat satellite dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    num_tiles, num_dates, num_bands = (
        len(data_set),
        get_satellite_dataset_size(data_set)[0],
        get_satellite_dataset_size(data_set)[1],
    )
    flattened_dataset_list, satellite_bands = (
        flatten_dataset_by_band(data_set),
        data_set["band"].values,
    )
    fig, ax = plt.subplots(
        num_bands, 1, figsize=(4, 4), squeeze=False
    )
    # --- start here ---

    # iterate by band
   
        # set scale to log
        
        # plot histogram
        
        # tick params on both axis with size of 9
        
        # set title w/ font size 9 to: "Landsat 8 Histogram (log scaled)\n{num_tiles} tiles, {num_dates} dates\nBand {satellite_bands[band]}"

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "landsat_histogram.png").touch(exist_ok=True)
        fig.savefig(image_dir / "landsat_histogram.png", format="png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()

    raise NotImplementedError

def plot_gt_histogram(
    data_set: xr.Dataset, image_dir: Path = None, n_bins: int = 4
) -> None:
    """
    This function plots the Ground Truth histogram over all Ground Truth values.

    Parameters
    ----------
    data_set: xr.Dataset
        The GT satellite dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    num_tiles = len(data_set)
    gt_flattened = flatten_dataset(data_set)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), squeeze=False, tight_layout=True)
    # --- start here ---

    # set scale to log
    
    # plot histogram
   
    # tick params on both axis with size of 9
    
    # set title w/ font size 9 to: "GT Histogram (log scaled), {num_tiles} tiles"
    

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        if not Path(Path(image_dir) / "ground_truth_histogram.png").exists():
            Path(Path(image_dir) / "ground_truth_histogram.png").touch(exist_ok=True)
        fig.savefig(Path(image_dir) / "ground_truth_histogram.png", format="png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()

    raise NotImplementedError

def plot_gt(data_array: xr.DataArray, image_dir: Path = None) -> None:
    """
    This function plots the Ground Truth image.

    Parameters
    ----------
    data_array: xr.DataArray
        A single tile from the GT dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), squeeze=False, tight_layout=True)
    # --- start here ---

    # imshow the ground truth image
    
    # set title to: "Ground Truth\n{data_array.attrs['parent_tile_id']}"
    
    # set the font size to 9
    

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "ground_truth.png").touch(exist_ok=True)
        fig.savefig(Path(image_dir) / "ground_truth.png", format="png")
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()
    
    raise NotImplementedError

def plot_max_projection_viirs(data_array: xr.DataArray, image_dir: Path = None) -> None:
    """
    This function plots the max projection VIIRS.

    Parameters
    ----------
    data_array: xr.DataArray
        The max projection of a single tile from the VIIRS dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), squeeze=False, tight_layout=True)
    # --- start here ---

    # show the image
    
    # set the title to: "Max Projection VIIRS\n{data_array.attrs['parent_tile_id']}"
   
    # set the font size to 9
   

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "viirs_max_projection.png").touch(exist_ok=True)
        fig.savefig(Path(image_dir) / "viirs_max_projection.png", format="png")
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()
    
    raise NotImplementedError

def plot_viirs(data_array: xr.DataArray, image_dir: Path = None) -> None:
    """
    This function plots the VIIRS image by date in subplots.

    Parameters
    ----------
    data_array: xr.DataArray
        A single tile from the VIIRS dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    num_dates = data_array.shape[0]
    num_rows, num_cols = 3, 3
    fig, ax = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4, 4),
        squeeze=False,
        tight_layout=True,
    )
    # --- start here ---

    # set the date index and band to 0, 0
    
    # iterate by rows
    
        # iterate by columns
       
            # if the date index is less that the num_dates
           
                # show the image
               
                # set title w/ font size 9 to: "VIIRS Image\n{data_array['date'][viirs_date_index].values}"
               
                # set the axis to off
                
            # else
            
                # set visibility to false
               
            # increment the date index
           

    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "viirs_plot_by_date.png").touch(exist_ok=True)
        fig.savefig(Path(image_dir) / "viirs_plot_by_date.png", format="png")
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()
    
    raise NotImplementedError

def create_rgb_composite_s1(data_array: xr.DataArray, image_dir: Path = None) -> None:
    """
    This function creates an RGB composite for Sentinel-1 using the formula:
        Red = VH band
        Green = VV band
        Blue = Red - Green

    Parameters
    ----------
    data_array: xr.DataArray
        A single tile from the S1 dataset
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.


    Returns
    -------
    None
    """
    num_dates = data_array.shape[0]
    num_rows, num_cols = 2, int(np.floor(np.divide(num_dates, 2)))
    fig, ax = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4, 4),
        squeeze=False,
        tight_layout=True,
    )
    # --- start here ---

    # set the date index to 0
    
    # iterate by rows
   
        # iterate by columns
        
            # if the date is less than num_dates
           
                # get Red by getting the VH image (.sel(band))
                
                # get Green by getting the VV image (.sel(band))
                
                # get Blue by doing (Red - Green) and clipping between 0 and 1
                
                # stack and show the image
                
                # set title w/ font size 9 to: "Sentinel 1 RGB composite\n{data_array[date]['date'].values}"
               
                # set the axis to off
                
            # else
            
                # set visibility to false
               
            # increment the date index
           
    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / "plot_sentinel1.png").touch(exist_ok=True)
        fig.savefig(image_dir / "plot_sentinel1.png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()
    
    raise NotImplementedError


def plot_satellite_by_bands(
    data_array: xr.DataArray, bands_to_plot: List[str], image_dir: Path = None
):
    """
    This function plots the satellite images by the bands to plot.

    Parameters
    ----------
    data_array: xr.DataArray
        A single tile from the satellite dataset
    bands_to_plot: List[str]
        A list of bands to stack and plot
    image_dir : Path
        The directory to save the image to. If none, the plot is just
        displayed
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    num_dates = data_array.shape[0]
    fig, ax = plt.subplots(
        num_dates,
        len(bands_to_plot),
        figsize=(4, 4),
        squeeze=False,
        tight_layout=True,
    )
    # --- start here ---

    # iterate through enumeration (index, bands) over the bands to plot
    
        # iterate by date
        
            # initalize a list to store the data
           
            # iterate over the bands
           
                # append the (date, band) image to the data list
                
            # stack and transpose the data list, then imshow
            
            # set the axis to off
            

            # set bands for title (empty string)
          
            # iterate over the bands
            
                # add the string to the bands for title string: band + " "
                
            # set title w/ font size 9 to: "{str(data_array.attrs['satellite_type']).capitalize()} {bands_for_title}\n{(data_array['date'])[date].values}"
            
    # --- end ---

    if image_dir is None:
        plt.show()
    elif isinstance(image_dir, Path):
        Path(image_dir / f"plot_{data_array.attrs['satellite_type']}.png").touch(
            exist_ok=True
        )
        fig.savefig(Path(image_dir) / f"plot_{data_array.attrs['satellite_type']}.png")
        plt.close()
    else:
        fig.savefig(image_dir, format="jpeg")
        plt.close()
    
    raise NotImplementedError