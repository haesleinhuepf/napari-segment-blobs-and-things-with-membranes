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
        invert_image

    image = np.asarray([[0, 1, 2, 3],
                        [2, 0, 1, 3],
                        [2, 0, 1, 3],
                        [2, 0, 1, 3]])

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
        seeded_watershed,
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
        thresholded_local_minima_seeded_watershed,
        sum_images,
        multiply_images,
        divide_images,
        invert_image]:

        print(operation)

        operation(image)
