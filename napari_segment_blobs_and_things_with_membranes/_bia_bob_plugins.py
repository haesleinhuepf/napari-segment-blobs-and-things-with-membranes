def list_bia_bob_plugins():
    """List of function hints for bia_bob"""
    good_alternative_installed = False

    try:
        import pyclesperanto
        good_alternative_installed = True
    except:
        pass
    try:
        import pyclesperanto_prototype
        good_alternative_installed = True
    except:
        pass

    basic_hints = ""
    if not good_alternative_installed:
        basic_hints = """
    
    * Applies Otsu's method to binarize an intensity image (also works with yen, isodata, li, mean, minimum, triangle instead of otsu).
      nsbatwm.threshold_otsu(image)
    
    * Segments blob-like structures using Voronoi-Otsu labeling.
      nsbatwm.voronoi_otsu_labeling(image, spot_sigma=4, outline_sigma=1)
    
    * Applies a Gaussian blur for noise reduction.
      nsbatwm.gaussian_blur(image, sigma=5)
      
    * Applies median filter to reduce noise while preserving edges.
      nsbatwm.median_filter(image, radius=5)
      
    * Smooth a label image using a local most popular intensity (mode) filter.
      nsbatwm.mode_filter(labels)
    
    * Removes background in an image using the top-hat filter.
      nsbatwm.white_tophat(image)
    
    * Applies local minimum filtering to an image (also works with maximum, and mean).
      nsbatwm.minimum_filter(image)
    
    * Subtracts background in an image using the rolling ball algorithm.
      nsbatwm.subtract_background(image)
    
    * Removes labeled objects touching image borders.
      nsbatwm.remove_labels_on_edges(label_image)
    
    * Expands labels by a specified distance.
      nsbatwm.expand_labels(label_image, distance=2)
    """

    return f"""    ## napari-segment-blobs-and-things-with-membranes (nsbatwm)
    nsbatwm is a Python library that processes images, mostly using the scikit-image library, but with simpler access.
    When you use it, you always start by importing the library: `import napari_segment_blobs_and_things_with_membranes as nsbatwm`.
    When asked for how to use nsbatwm, you can adapt one of the following code snippets:
    
    {basic_hints}
  
    * Splits touching objects in a binary image using an algorithm similar to the ImageJ watershed.
      nsbatwm.split_touching_objects(binary_image)
      
    * Labels connected components in a binary image.
      nsbatwm.connected_component_labeling(binary_image)
      
    * Applies seeded watershed segmentation using labeled objects, e.g. nuclei, and an image showing bright borders between objects such as cell membranes.
      nsbatwm.seeded_watershed(image, labeled_objects)
      
    * Applies a percentile filter.
      nsbatwm.percentile_filter(image)
    
    * Segments using seeded watershed with local minima as seeds.
      nsbatwm.local_minima_seeded_watershed(image, spot_sigma=10, outline_sigma=2)
    
    * Skeletonizes labeled objects.
      nsbatwm.skeletonize(image)
    """