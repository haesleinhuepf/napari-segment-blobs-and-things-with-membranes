# from napari_segment_blobs_and_things_with_membranes import threshold, image_arithmetic

# add your tests here...
import numpy as np


def test_something():
    from napari_segment_blobs_and_things_with_membranes import gaussian_blur, \
        subtract_background,\
        threshold_otsu,\
        threshold_yen,\
        threshold_isodata,\
        threshold_li,\
        threshold_mean,\
        threshold_minimum,\
        threshold_triangle,\
        binary_invert,\
        split_touching_objects,\
        connected_component_labeling,\
        seeded_watershed, \
        seeded_watershed_with_mask, \
        voronoi_otsu_labeling, \
        gauss_otsu_labeling,\
        gaussian_laplace,\
        median_filter, \
        mode_filter, \
        maximum_filter,\
        minimum_filter,\
        percentile_filter,\
        black_tophat,\
        white_tophat,\
        morphological_gradient,\
        local_minima_seeded_watershed,\
        thresholded_local_minima_seeded_watershed,\
        sum_images,\
        multiply_images,\
        divide_images,\
        invert_image, \
        skeletonize, \
        rescale, \
        resize, \
        butterworth, \
        extract_slice, \
        sub_sample, \
        squeeze, \
        grayscale_erosion, \
        binary_erosion, \
        grayscale_dilation, \
        binary_dilation, \
        grayscale_opening, \
        binary_opening, \
        grayscale_closing, \
        binary_closing

    import numpy as np

    image_2d = np.asarray([[0, 1, 2, 3],
                           [2, 0, 1, 3],
                           [2, 253, 1, 3],
                           [255, 253, 1, 3]])

    image_3d = np.ones((4, 4, 4))

    for operation in [gaussian_blur,
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
        invert_image,
        rescale,
        resize,
        butterworth,
        extract_slice,
        sub_sample,
        squeeze]:

        print(operation)

        operation(image_2d)

    for operation in [
        seeded_watershed,
        sum_images,
        multiply_images,
        divide_images]:

        print(operation)

        operation(image_2d, image_2d)

    for operation in [
        grayscale_erosion,
        binary_erosion,
        grayscale_dilation,
        binary_dilation,
        grayscale_opening,
        binary_opening,
        grayscale_closing,
        binary_closing
    ]:
        for image in (image_2d, image_3d):
            labels = image > 0
            print(f"{operation} with {labels.ndim}d image")
            operation(labels)

    skeletonize(image_2d > 0)

    seeded_watershed_with_mask(image_2d, image_2d, image_2d)

    mode_filter(image_2d.astype(int))

def test_remove_labels_on_edges_sequential_labeling():
    image = np.asarray([
        [1,2,3],
        [4,5,6],
        [7,7,7],
    ])

    reference = np.asarray([
        [0,0,0],
        [0,1,0],
        [0,0,0],
    ])

    from napari_segment_blobs_and_things_with_membranes import remove_labels_on_edges
    result = remove_labels_on_edges(image)

    print(result)
    print(reference)

    assert np.array_equal(result, reference)

def test_connected_component_labeling_sequential_labeling():
    image = np.asarray([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
    ])

    reference = np.asarray([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])

    from napari_segment_blobs_and_things_with_membranes import connected_component_labeling
    result = connected_component_labeling(image, exclude_on_edges=True)

    print(result)
    print(reference)

    assert np.array_equal(result, reference)


def test_relabel_sequential():
    image = np.asarray([[0, 0, 0, 1, 3, 4]])
    reference = np.asarray([[0, 0, 0, 1, 2, 3]])

    from napari_segment_blobs_and_things_with_membranes import relabel_sequential
    result = relabel_sequential(image)

    print(result)
    print(reference)

    assert np.array_equal(result, reference)

