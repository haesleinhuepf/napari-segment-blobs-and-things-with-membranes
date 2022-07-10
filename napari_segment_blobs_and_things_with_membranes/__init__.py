
__version__ = "0.2.22"
__common_alias__ = "nsbatwm"

from napari.types import ImageData, LabelsData

from napari_plugin_engine import napari_hook_implementation

import numpy as np
from scipy import ndimage as ndi

from skimage.filters import threshold_otsu as sk_threshold_otsu, gaussian, sobel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import binary_opening
from skimage.measure import label
from skimage.morphology import local_maxima, local_minima
from skimage.restoration import rolling_ball
from napari_tools_menu import register_function
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from skimage.segmentation import clear_border
from skimage.segmentation import expand_labels as sk_expand_labels

from skimage import filters
import scipy
from scipy import ndimage
import napari
from napari_time_slicer import time_slicer

@napari_hook_implementation
def napari_experimental_provide_function():
    return [
        gaussian_blur,
        subtract_background,
        threshold_otsu,
        threshold_yen,
        threshold_isodata,
        threshold_li,
        threshold_mean,
        threshold_minimum,
        threshold_triangle,
        binary_invert,
        split_touching_objects,
        connected_component_labeling,
        seeded_watershed,
        voronoi_otsu_labeling,
        gauss_otsu_labeling,
        gaussian_laplace,
        median_filter,
        maximum_filter,
        minimum_filter,
        percentile_filter,
        black_tophat,
        white_tophat,
        morphological_gradient,
        local_minima_seeded_watershed,
        thresholded_local_minima_seeded_watershed,
        sum_images,
        multiply_images,
        divide_images,
        invert_image,
        skeletonize,
        Manually_merge_labels,
        Manually_split_labels
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


@register_function(menu="Segmentation post-processing > Split touching objects (nsbatwm)")
@time_slicer
def split_touching_objects(binary:LabelsData, sigma: float = 3.5) -> LabelsData:
    """
    Takes a binary image and draws cuts in the objects similar to the ImageJ watershed algorithm [1].

    This allows cutting connected objects such as not to dense nuclei. If the nuclei are too dense,
    consider using stardist [2] or cellpose [3].

    See also
    --------
    .. [1] https://imagej.nih.gov/ij/docs/menus/process.html#watershed
    .. [2] https://www.napari-hub.org/plugins/stardist-napari
    .. [3] https://www.napari-hub.org/plugins/cellpose-napari
    """
    binary = np.asarray(binary)

    # typical way of using scikit-image watershed
    distance = ndi.distance_transform_edt(binary)
    blurred_distance = gaussian(distance, sigma=sigma)
    fp = np.ones((3,) * binary.ndim)
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


@register_function(menu="Segmentation / binarization > Threshold (Otsu et al 1979, scikit-image, nsbatwm)")
@time_slicer
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


@register_function(menu="Segmentation / binarization > Threshold (Yen et al 1995, scikit-image, nsbatwm)")
@time_slicer
def threshold_yen(image : napari.types.ImageData) -> napari.types.LabelsData:
    """
    Binarize an image according to Yen's method.

    Parameters
    ----------
    image: Image

    See Also
    --------
    ..[0] https://imagej.net/plugins/auto-threshold#yen

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    return image > filters.threshold_yen(image)


@register_function(menu="Segmentation / binarization > Threshold (Isodata, Ridler et al 1978, scikit-image, nsbatwm)")
@time_slicer
def threshold_isodata(image : napari.types.ImageData) -> napari.types.LabelsData:
    """
    Binarize an image according to the IsoData / Ridler's method.
    The method is similar to ImageJ's "default" threshold.

    Parameters
    ----------
    image: Image

    See Also
    --------
    ..[0] https://imagej.net/plugins/auto-threshold#isodata
    ..[1] https://ieeexplore.ieee.org/document/4310039

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    return image > filters.threshold_isodata(image)


@register_function(menu="Segmentation / binarization > Threshold (Li et al 1993, scikit-image, nsbatwm)")
@time_slicer
def threshold_li(image : napari.types.ImageData) -> napari.types.LabelsData:
    """
    Binarize an image according to Li's method method.

    Parameters
    ----------
    image: Image

    See Also
    --------
    ..[0] https://imagej.net/plugins/auto-threshold#li

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    return image > filters.threshold_li(image)


@register_function(menu="Segmentation / binarization > Threshold (Ridler et al 1978, scikit-image, nsbatwm)")
@time_slicer
def threshold_mean(image : napari.types.ImageData) -> napari.types.LabelsData:
    """
    Binarize an image according to the Mean method.

    Parameters
    ----------
    image: Image

    See Also
    --------
    ..[0] https://imagej.net/plugins/auto-threshold#mean

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    return image > filters.threshold_mean(image)


@register_function(menu="Segmentation / binarization > Threshold (Mean, scikit-image, nsbatwm)")
@time_slicer
def threshold_minimum(image : napari.types.ImageData) -> napari.types.LabelsData:
    """
    Binarize an image according to the Minimum method.

    Parameters
    ----------
    image: Image

    See Also
    --------
    ..[0] https://imagej.net/plugins/auto-threshold#minimum

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    return image > filters.threshold_minimum(image)


@register_function(menu="Segmentation / binarization > Threshold (Triangle method, Zack et al 1977, scikit-image, nsbatwm)")
@time_slicer
def threshold_triangle(image : napari.types.ImageData) -> napari.types.LabelsData:
    """
    Binarize an image according to the Triangle method.

    Parameters
    ----------
    image: Image

    See Also
    --------
    ..[0] https://imagej.net/plugins/auto-threshold#triangle

    Returns
    -------
    binary_image: napari.types.LabelsData
    """
    return image > filters.threshold_triangle(image)


@register_function(menu="Filtering / noise removal > Gaussian (scikit-image, nsbatwm)")
@time_slicer
def gaussian_blur(image:ImageData, sigma: float = 1) -> ImageData:
    """
    Applies a Gaussian blur to an image with a defined sigma. Useful for denoising.
    """
    return gaussian(image, sigma)


@register_function(menu="Filtering / edge enhancement > Gaussian Laplace (scipy, nsbatwm)")
@time_slicer
def gaussian_laplace(image: napari.types.ImageData, sigma: float = 2) -> napari.types.ImageData:
    """
    Apply Laplace filter for edge detection / edge enhancement after applying a Gaussian-blur

    Parameters
    ----------
    image: array-like
        Image to detect edges in
    sigma: float
        The filter will be applied with this specified Gaussian-blur sigma

    Returns
    -------
    array-like

    See also
    --------
    .. [1] https://en.wikipedia.org/wiki/Laplace_operator
    """
    return scipy.ndimage.gaussian_laplace(image.astype(float), sigma)


@register_function(menu="Filtering / noise removal > Median (scipy, nsbatwm)")
@time_slicer
def median_filter(image: napari.types.ImageData, radius: float = 2) -> napari.types.ImageData:
    """
    The median-filter allows removing noise from images. While locally averaging intensity, it
    is an edge-preserving filter.

    It is equal to a percentile-filter with percentile==50.
    In case applying the filter takes to much time, consider using a Gaussian blur instead.
    """
    return scipy.ndimage.median_filter(image.astype(float), size=int(radius * 2 + 1))


@register_function(menu="Filtering / noise removal > Percentile (scipy, nsbatwm)")
@time_slicer
def percentile_filter(image: napari.types.ImageData, percentile : float = 50, radius: float = 2) -> napari.types.ImageData:
    """The percentile filter is similar to the median-filter but it allows specifying the percentile.
    The percentile-filter with percentile==50 is equal to the median-filter.
    """
    return scipy.ndimage.percentile_filter(image.astype(float), percentile=percentile, size=int(radius * 2 + 1))


@register_function(menu="Filtering / background removal > White top-hat (scipy, nsbatwm)")
@time_slicer
def white_tophat(image: napari.types.ImageData, radius: float = 2) -> napari.types.ImageData:
    """
    The white top-hat filter removes bright regions from an image showing black islands.

    In the context of fluorescence microscopy, it allows removing intensity resulting from out-of-focus light.
    """
    return scipy.ndimage.white_tophat(image.astype(float), size=int(radius * 2 + 1))


@register_function(menu="Filtering / background removal > Black top-hat (scipy, nsbatwm)")
@time_slicer
def black_tophat(image: napari.types.ImageData, radius: float = 2) -> napari.types.ImageData:
    """
    The black top-hat filter removes bright regions from an image showing black islands.
    """
    return scipy.ndimage.black_tophat(image.astype(float), size=int(radius * 2 + 1))


@register_function(menu="Filtering / background removal > Minimum (scipy, nsbatwm)")
@time_slicer
def minimum_filter(image: napari.types.ImageData, radius: float = 2) -> napari.types.ImageData:
    """
    Local minimum filter

    Can be used for noise and background removal.
    """
    return scipy.ndimage.minimum_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / background removal > Maximum (scipy, nsbatwm)")
@time_slicer
def maximum_filter(image: napari.types.ImageData, radius: float = 2) -> napari.types.ImageData:
    """
    Local maximum filter

    In the context of cell segmentation it can be used to make membranes wider
    and close small gaps of insufficient staining.
    """
    return scipy.ndimage.maximum_filter(image.astype(float), size=radius * 2 + 1)


@register_function(menu="Filtering / edge enhancement > Morphological Gradient (scipy, nsbatwm)")
@time_slicer
def morphological_gradient(image: napari.types.ImageData, radius: float = 2) -> napari.types.ImageData:
    """
    Apply gradient filter (similar to the Sobel operator) for edge detection / edge enhancement.
    This is similar to applying a Gaussian-blur to an image and afterwards the gradient operator

    Parameters
    ----------
    image: array-like
        Image to detect edges in
    radius: float
        The filter will be applied with a kernel size of (radius * 2 + 1)

    Returns
    -------
    array-like

    See also
    --------
    .. [1] https://en.wikipedia.org/wiki/Morphological_gradient
    """
    return scipy.ndimage.morphological_gradient(image.astype(float), size=int(radius * 2 + 1))


@register_function(menu="Filtering / background removal > Rolling ball (scikit-image, nsbatwm)")
@time_slicer
def subtract_background(image:ImageData, rolling_ball_radius: float = 5) -> ImageData:
    """
    Subtract background in an image using the rolling-ball algorithm.

    See also
    --------
    ..[0] https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rolling_ball.html
    """
    background = rolling_ball(image, radius = rolling_ball_radius)
    return image - background


@register_function(menu="Segmentation post-processing > Invert binary image (nsbatwm)")
@time_slicer
def binary_invert(binary_image:LabelsData) -> LabelsData:
    """
    Inverts a binary image.
    """
    return (np.asarray(binary_image) == 0) * 1


@register_function(menu="Segmentation / labeling > Connected component labeling (scikit-image, nsbatwm)")
@time_slicer
def connected_component_labeling(binary_image: LabelsData, exclude_on_edges: bool = False,
                                 viewer: napari.Viewer = None) -> LabelsData:
    """
    Takes a binary image and produces a label image with all separated objects labeled with
    different integer numbers.

    Parameters
    ----------
    exclude_on_edges : bool, optional
        Whether or not to clear objects connected to the label image border/planes (either xy, xz or yz).
    """
    if exclude_on_edges:
        # processing the image, which is not a timelapse
        return remove_labels_on_edges(label(np.asarray(binary_image)))

    else:
        return label(np.asarray(binary_image))


@register_function(menu="Segmentation post-processing > Remove labeled objects at the image border (scikit-image, nsbatwm)")
@time_slicer
def remove_labels_on_edges(label_image: LabelsData) -> LabelsData:
    """
    Takes a label image and removes objects that touch the image border.
    The remaining labels are relabeled sequentially.

    See also
    --------
    ..[0] https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.clear_border
    """

    result = clear_border(np.asarray(label_image))
    relabeled_result, _, _ = relabel_sequential(result)
    return relabeled_result


@register_function(menu="Segmentation post-processing > Expand labels (scikit-image, nsbatwm)")
@time_slicer
def expand_labels(label_image: LabelsData, distance: float = 1) -> LabelsData:
    """
    Takes a label image and makes labels larger up to a given radius (distance).
    Labels will not overwrite each other while expanding. This operation is also known as label dilation.

    See also
    --------
    ..[0] https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.expand_labels
    """

    return sk_expand_labels(np.asarray(label_image), distance=distance)


@register_function(menu="Segmentation / labeling > Voronoi-Otsu-labeling (nsbatwm)")
@time_slicer
def voronoi_otsu_labeling(image:ImageData, spot_sigma: float = 2, outline_sigma: float = 2) -> LabelsData:
    """Voronoi-Otsu-Labeling is a segmentation algorithm for blob-like structures such as nuclei and
    granules with high signal intensity on low-intensity background.

    The two sigma parameters allow tuning the segmentation result. The first sigma controls how close detected cells
    can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
    hood, this filter applies two Gaussian blurs, spot detection, Otsu-thresholding and Voronoi-labeling. The
    thresholded binary image is flooded using the Voronoi approach starting from the found local maxima. Noise-removal
    sigma for spot detection and thresholding can be configured separately.

    This allows segmenting connected objects such as not to dense nuclei.
    If the nuclei are too dense, consider using stardist [1] or cellpose [2].

    See also
    --------
    .. [0] https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb
    .. [1] https://www.napari-hub.org/plugins/stardist-napari
    .. [2] https://www.napari-hub.org/plugins/cellpose-napari
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


@register_function(menu="Segmentation / labeling > Gauss-Otsu-labeling (nsbatwm)")
@time_slicer
def gauss_otsu_labeling(image:ImageData, outline_sigma: float = 2) -> LabelsData:
    """Gauss-Otsu-Labeling can be used to segment objects such as nuclei with bright intensity on
    low intensity background images.

    The outline_sigma parameter allows tuning how precise segmented objects are outlined. Under the
    hood, this filter applies a Gaussian blur, Otsu-thresholding and connected component labeling.

    See also
    --------
    .. [0] https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/gauss_otsu_labeling.ipynb
    """
    image = np.asarray(image)

    # blur
    blurred_outline = gaussian(image, outline_sigma)

    # threshold
    threshold = sk_threshold_otsu(blurred_outline)
    binary_otsu = blurred_outline > threshold

    # connected component labeling
    labels = label(binary_otsu)

    return labels


@register_function(menu="Segmentation post-processing > Seeded watershed (scikit-image, nsbatwm)")
@time_slicer
def seeded_watershed(membranes:ImageData, labeled_nuclei:LabelsData) -> LabelsData:
    """
    Takes a image with bright (high intensity) membranes and an image with labeled objects such as nuclei.
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


@register_function(menu="Segmentation / labeling > Seeded watershed using local minima as seeds (nsbatwm)")
@time_slicer
def local_minima_seeded_watershed(image:ImageData, spot_sigma: float = 10, outline_sigma: float = 0) -> LabelsData:
    """
    Segment cells in images with fluorescently marked membranes.

    The two sigma parameters allow tuning the segmentation result. The first sigma controls how close detected cells
    can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
    hood, this filter applies two Gaussian blurs, local minima detection and a seeded watershed.

    See also
    --------
    .. [1] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    """

    image = np.asarray(image)

    spot_blurred = gaussian(image, sigma=spot_sigma)

    spots = label(local_minima(spot_blurred))

    if outline_sigma == spot_sigma:
        outline_blurred = spot_blurred
    else:
        outline_blurred = gaussian(image, sigma=outline_sigma)

    return watershed(outline_blurred, spots)


@register_function(menu="Segmentation / labeling > Seeded watershed using local minima as seeds and an intensity threshold (nsbatwm)")
@time_slicer
def thresholded_local_minima_seeded_watershed(image:ImageData, spot_sigma: float = 3, outline_sigma: float = 0, minimum_intensity: float = 500) -> LabelsData:
    """
    Segment cells in images with marked membranes that have a high signal intensity.

    The two sigma parameters allow tuning the segmentation result. The first sigma controls how close detected cells
    can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
    hood, this filter applies two Gaussian blurs, local minima detection and a seeded watershed.

    Afterwards, all objects are removed that have an average intensity below a given minimum_intensity
    """
    labels = local_minima_seeded_watershed(image, spot_sigma=spot_sigma, outline_sigma=outline_sigma)

    # measure intensities
    stats = regionprops(labels, image)
    intensities = [r.mean_intensity for r in stats]

    # filter labels with low intensity
    new_label_indices, _, _ = relabel_sequential((np.asarray(intensities) > minimum_intensity) * np.arange(labels.max()))
    new_label_indices = np.insert(new_label_indices, 0, 0)
    new_labels = np.take(np.asarray(new_label_indices, np.uint32), labels)

    return new_labels

@register_function(menu="Image math > Sum images (numpy, nsbatwm)", factor1={'min': -1000000, 'max': 1000000}, factor2={'min': -1000000, 'max': 1000000})
@time_slicer
def sum_images(image1: ImageData, image2: ImageData, factor1: float = 1, factor2: float = 1) -> ImageData:
    """Add two images"""
    return image1 * factor1 + image2 * factor2


@register_function(menu="Image math > Multiply images (numpy, nsbatwm)")
@time_slicer
def multiply_images(image1: ImageData, image2: ImageData) -> ImageData:
    """Multiply two images"""
    return image1 * image2


@register_function(menu="Image math > Divide images (numpy, nsbatwm)")
@time_slicer
def divide_images(image1: ImageData, image2: ImageData) -> ImageData:
    """Divide one image by another"""
    return image1 / image2


@register_function(menu="Image math > Invert image (scikit-image, nsbatwm)")
@time_slicer
def invert_image(image: ImageData) -> ImageData:
    """Invert an image. The excat math behind depends on the image type.

    See also
    --------
    .. [0] https://scikit-image.org/docs/dev/api/skimage.util.html?highlight=invert#skimage.util.invert
    """
    from skimage import util
    return util.invert(image)


@register_function(menu="Segmentation post-processing > Skeletonize (scikit-image, nsbatwm)")
@time_slicer
def skeletonize(image: LabelsData) -> LabelsData:
    """
    Skeletonize labeled objects in an image. This can be useful to reduce objects such as neurons, roots and vessels
    with variable thickness to single pixel lines for further analysis.

    See also
    --------
    .. [0] https://en.wikipedia.org/wiki/Topological_skeleton
    """
    from skimage import morphology
    if image.max() == 1:
        return morphology.skeletonize(image)
    else:
        result = np.zeros(image.shape)
        for i in range(1, image.max() + 1):
            skeleton = morphology.skeletonize(image == i)
            result = skeleton * i + result
        return result.astype(int)


@register_function(menu="Utilities > Manually merge labels (nsbatwm)")
def Manually_merge_labels(labels_layer: napari.layers.Labels, points_layer: napari.layers.Points, viewer : napari.Viewer):
    if points_layer is None:
        points_layer = viewer.add_points([])
        points_layer.mode = 'ADD'
        return
    labels = np.asarray(labels_layer.data)
    points = points_layer.data

    label_ids = [labels.item(tuple([int(j) for j in i])) for i in points]

    # replace labels with minimum of the selected labels
    new_label_id = min(label_ids)
    for l in label_ids:
        if l != new_label_id:
            labels[labels == l] = new_label_id

    labels_layer.data = labels
    points_layer.data = []

@register_function(menu="Utilities > Manually split labels (nsbatwm)")
def Manually_split_labels(labels_layer: napari.layers.Labels, points_layer: napari.layers.Points, viewer: napari.Viewer):
    if points_layer is None:
        points_layer = viewer.add_points([])
        points_layer.mode = 'ADD'
        return

    labels = np.asarray(labels_layer.data)
    points = points_layer.data

    label_ids = [labels.item(tuple([int(j) for j in i])) for i in points]

    # make a binary image first
    binary = np.zeros(labels.shape, dtype=bool)
    new_label_id = min(label_ids)
    for l in label_ids:
        binary[labels == l] = True

    # origin: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    #from skimage.feature import peak_local_max

    #distance = ndi.distance_transform_edt(binary)
    #coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
    mask = np.zeros(labels.shape, dtype=bool)
    for i in points:
        #mask[tuple(points)] = True
        mask[tuple([int(j) for j in i])] = True

    markers, _ = ndi.label(mask)
    new_labels = watershed(binary, markers, mask=binary)
    labels[binary] = new_labels[binary] + labels.max()

    labels_layer.data = labels
    points_layer.data = []


