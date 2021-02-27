# std
from typing import Tuple, Optional, List

# lib
import numpy as np  # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from IPython.display import set_matplotlib_formats # type: ignore


def set_style() -> None:
    """Set some styling preferences for the notebook"""
    np.set_printoptions(suppress=True, precision=3,
                        threshold=10000, linewidth=70,
                        formatter={'float': lambda x: ' {:.3}'.format(x)})
    np.set_printoptions(suppress=True)
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'figure.titlesize': 18})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'axes.labelsize': 20})
    sns.set_style("whitegrid")


def display_images(images_1d: List[np.ndarray],
                   shape: Tuple[int, int] = (2, 2),
                   titles: Optional[List[str]] = None,
                   annot: bool = True) -> None:
    """ Compact plotting of multiple 1D image np.arrays with title annotations

    images_1d  -- List of 1-D shaped numpy arrays with the image intensities
    shape      -- tuple with dimensions (width and height) of the images
                  It's expected that all images have the same dimension
    title      -- Optional list of annotations for the respective images
    """
    assert titles is None or len(titles) == len(images_1d), \
            "Non matching number of titles and images"

    num_images = len(images_1d)
    f, axes = plt.subplots(1, num_images, figsize=(num_images * 3 + 3,3))

    for i in range(num_images):
        image_2d = images_1d[i].reshape(shape)
        ax = axes[i] if num_images > 1 else axes
        sns.heatmap(image_2d, cmap="gray", vmin=0.0, vmax=255.0, fmt=".1f", annot=annot, cbar=annot, square=True, ax=ax)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i], pad=20)

    f.tight_layout()

def display_valid_observations(images: np.ndarray, low_limit: int = 10, high_limit: int = 250) -> None:
    """Display statistics about valid pixel observations in an array of images

    Default limits are set to ignore very low and high values close to the over- / underexposure.

    images     -- 2D array with 1D images
    low_limit  -- limit at which valid observations are cut off
    high_limit -- limit at which valid observations are cut off
    """
    valid_count = ((low_limit < images) & (images < high_limit)).sum(axis=0)

    f, ax = plt.subplots()
    ax.set_title("Valid observations in range [{}, {}]".format(low_limit, high_limit))
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Pixel ID")
    ax.bar(range(images.shape[1]), valid_count)

def display_observations_distribution(images: np.ndarray) -> None:
    """Display bar chart with distribution of valid pixel observations in the images

    images     -- 2D array with 1D images
    """
    assert len(images.shape) == 2, "Unexpected input array shape"

    [num_images, image_size] = images.shape
    assert num_images > 0, "At least one image is required"
    assert 0 < image_size <= 4, "The range of supported image sizes is 1-4 pixels"

    value_range = range(1,255)
    f, axes = plt.subplots(image_size, 1, figsize=(12, image_size * 3))

    pixel_stats = []
    for pix_idx in range(image_size):
        value_counts = []
        for value in value_range:
            value_counts.append((images[:, pix_idx] == value).sum())
        pixel_stats.append(value_counts)

    for pix_idx in range(image_size):
        ax = axes[pix_idx] if image_size > 1 else axes
        ax.set_title("Stats for pixel ID: {}".format(pix_idx))
        ax.set_ylabel("Number of images")
        ax.set_xlabel("Pixel Value")
        ax.bar(value_range, pixel_stats[pix_idx])

    f.tight_layout()


def plot(ax, xs: np.ndarray, ys: np.ndarray, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(xs, ys)

def plot_exposures(exposures: List[float]) -> None:
    f, ax = plt.subplots()
    plt.title("Exposure time (seconds)")
    plt.xlabel("Sequence number")
    plt.ylabel("Value in seconds")
    plt.plot(exposures)

def plot_inverse_crf_side_by_side(U1: np.ndarray, U2: np.ndarray) -> None:
    f, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_inverse_crf(U1, ax = axes[0])
    plot_inverse_crf(U2, ax = axes[1])

    f.tight_layout()


def plot_inverse_crf(U: np.ndarray, ax = None) -> None:
    if ax is None:
        f, ax = plt.subplots(1, 1)

    ax.set_title("Inverse CRF")
    ax.set_xlabel("Pixel values")
    ax.set_ylabel("Relative irradiance")
    ax.plot(U)

def plot_multiple_inverse_crf(U_list: List[np.ndarray]) -> None:
    """ Compact plotting of multiple 1D image np.arrays with title annotations

    U -- List of reverse CRF functions to plot
    """
    num_functions = len(U_list)
    f, axes = plt.subplots(1, num_functions, figsize=(num_functions * 3 + 3,3))

    for i in range(num_functions):
        axes[i].plot(U_list[i])

    f.tight_layout()

def print_error_to_ground_truth(estimate: np.ndarray, ground_truth: np.ndarray) -> None:
    print("Ground-truth: ", ground_truth)
    print("Estimated: ", estimate)
    print("Error (diff): ", ground_truth - estimate)

def show_experiment_errors(estimates: List[np.ndarray], ground_truth: np.ndarray) -> None:
    rmse = np.sqrt(np.mean((estimates-ground_truth)**2))
    print("RMSE: ", rmse)

    f, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_per_pixel_rmse(estimates, ground_truth, ax = axes[0])
    plot_diff_errors_to_ground_truth(estimates, ground_truth, ax = axes[1])
    f.tight_layout()

def plot_per_pixel_rmse(estimates: List[np.ndarray], ground_truth: np.ndarray, ax = None) -> None:
    rmse = np.sqrt(np.mean((estimates-ground_truth)**2, axis=0))

    if ax is None:
        f, ax = plt.subplots()

    ax.set_title("Per Pixel RMSE")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Pixel ID")
    ax.bar(range(estimates.shape[1]), rmse)

def plot_diff_errors_to_ground_truth(estimates: List[np.ndarray], ground_truth: np.ndarray, ax = None) -> None:
    errors = []
    for estimate in estimates:
        errors.append((ground_truth - estimate))

    if ax is None:
        f, ax = plt.subplots()

    ax.set_title("Estimation errors (diff)")
    ax.set_xlabel("Experiment ID")
    ax.set_ylabel("Error")
    ax.plot(errors)

def get_irradiance_estimates(images_per_exposure: int, number_experiments: int) -> List[np.ndarray]:
    estimates = []

    for _ in range(number_experiments):
        images, exposure_times = generate_sliding_exposure_dataset(
            E=get_extreme_sensor_irradiance(), G=linear_crf)
        irradiance_estimate = recover_sensor_irradiance_average(images, exposure_times, U=linear_inverse_crf)
        estimates.append(irradiance_estimate)

    return estimates
