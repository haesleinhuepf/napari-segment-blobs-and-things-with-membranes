# from napari_segment_blobs_and_things_with_membranes import threshold, image_arithmetic

# add your tests here...


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
        seeded_watershed,\
        voronoi_otsu_labeling,\
        gaussian_laplace,\
        median_filter,\
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
        skeletonize

    import numpy as np

    image = np.asarray([[0, 1, 2, 3],
                        [2, 0, 1, 3],
                        [2, 253, 1, 3],
                        [255, 253, 1, 3]])

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
        gaussian_laplace,
        median_filter,
        maximum_filter,
        minimum_filter,
        percentile_filter,
        black_tophat,
        white_tophat,
        morphological_gradient,
        local_minima_seeded_watershed,
        invert_image]:

        print(operation)

        operation(image)

    for operation in [
        seeded_watershed,
        sum_images,
        multiply_images,
        divide_images]:

        print(operation)

        operation(image, image)

    skeletonize(image > 0)
