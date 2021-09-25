from napari.types import ImageData, LabelsData

from napari_plugin_engine import napari_hook_implementation

import numpy as np
from scipy import ndimage as ndi

from skimage.filters import threshold_otsu as sk_threshold_otsu, gaussian, sobel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import binary_opening
from skimage.measure import label
from skimage.morphology import local_maxima
from skimage.restoration import rolling_ball

@napari_hook_implementation
def napari_experimental_provide_function():
    return [
        gaussian_blur,
        subtract_background,
        threshold_otsu,
        binary_invert,
        split_touching_objects,
        connected_components_labeling,
        seeded_watershed,
        voronoi_otsu_labeling
    ]

def _sobel_3d(image):
    kernel = np.asarray([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], [
            [0, 1, 0],
            [1, -6, 1],
            [0, 1, 0]
        ], [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
    ])
    return ndi.convolve(image, kernel)

def split_touching_objects(binary:LabelsData, sigma:float=3.5) -> LabelsData:
    """
    Takes a binary image and draws cuts in the objects similar to the ImageJ watershed algorithm.

    See also
    --------
    .. [0] https://imagej.nih.gov/ij/docs/menus/process.html#watershed
    """
    binary = np.asarray(binary)

    # typical way of using scikit-image watershed
    distance = ndi.distance_transform_edt(binary)
    blurred_distance = gaussian(distance, sigma=sigma)
    fp = np.ones((3, 3)) if len(binary.shape) == 2 else np.ones((3, 3, 3))
    coords = peak_local_max(blurred_distance, footprint=fp, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    labels = watershed(-blurred_distance, markers, mask=binary)

    # identify label-cutting edges
    if len(binary.shape) == 2:
        edges = sobel(labels)
        edges2 = sobel(binary)
    else: # assuming 3D
        edges = _sobel_3d(labels)
        edges2 = _sobel_3d(binary)

    almost = np.logical_not(np.logical_xor(edges != 0, edges2 != 0)) * binary
    return binary_opening(almost)


def threshold_otsu(image:ImageData) -> LabelsData:
    """
    Applies Otsu's threshold selection method to an intensity image and returns a binary image with pixels==1 where
    intensity is above the determined threshold.

    See also
    --------
    .. [0] https://en.wikipedia.org/wiki/Otsu%27s_method
    .. [1] https://ieeexplore.ieee.org/document/4310076
    """
    threshold = sk_threshold_otsu(np.asarray(image))
    binary_otsu = image > threshold

    return binary_otsu * 1

def gaussian_blur(image:ImageData, sigma:float=1) -> ImageData:
    """
    Applies a Gaussian blur to an image with a defined sigma. Useful for denoising.
    """
    return gaussian(image, sigma)

def subtract_background(image:ImageData, rolling_ball_radius:float = 5) -> ImageData:
    background = rolling_ball(image, radius = rolling_ball_radius)
    return image - background


def binary_invert(binary_image:LabelsData) -> LabelsData:
    """
    Inverts a binary image.
    """
    return (np.asarray(binary_image) == 0) * 1

def connected_components_labeling(binary_image:LabelsData) -> LabelsData:
    """
    Takes a binary image and produces a label image with all separated objects labeled differently.
    """
    return label(np.asarray(binary_image))


def voronoi_otsu_labeling(image:ImageData, spot_sigma: float = 2, outline_sigma: float = 2) -> LabelsData:
    """
    The two sigma parameters allow tuning the segmentation result. The first sigma controls how close detected cells
    can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
    hood, this filter applies two Gaussian blurs, spot detection, Otsu-thresholding and Voronoi-labeling. The
    thresholded binary image is flooded using the Voronoi approach starting from the found local maxima. Noise-removal
    sigma for spot detection and thresholding can be configured separately.

    See also
    --------
    .. [0] https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb
    """
    image = np.asarray(image)

    # blur and detect local maxima
    blurred_spots = gaussian(image, spot_sigma)
    spot_centroids = local_maxima(blurred_spots)

    # blur and threshold
    blurred_outline = gaussian(image, outline_sigma)
    threshold = sk_threshold_otsu(blurred_outline)
    binary_otsu = blurred_outline > threshold

    # determine local maxima within the thresholded area
    remaining_spots = spot_centroids * binary_otsu

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = watershed(binary_otsu, labeled_spots, mask=binary_otsu)

    return labels

def seeded_watershed(membranes:ImageData, labeled_nuclei:LabelsData) -> LabelsData:
    """
    Takes a image with brigh (high intensity) membranes and an image with labeled objects such as nuclei.
    The latter serves as seeds image for a watershed labeling.

    See also
    --------
    .. [1] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    """
    cells = watershed(
        np.asarray(membranes),
        np.asarray(labeled_nuclei)
    )
    return cells