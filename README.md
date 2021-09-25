# napari-segment-blobs-and-things-with-membranes

[![License](https://img.shields.io/pypi/l/napari-segment-blobs-and-things-with-membranes.svg?color=green)](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-segment-blobs-and-things-with-membranes.svg?color=green)](https://pypi.org/project/napari-segment-blobs-and-things-with-membranes)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-segment-blobs-and-things-with-membranes.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/branch/master/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-segment-blobs-and-things-with-membranes)

A plugin based on scikit-image for segmenting nuclei and cells based on fluorescent microscopy images with high intensity in nuclei and/or membranes. 
The available functions and their user interface based on [magicgui](https://napari.org/magicgui/) are shown below. You can also call these functions
as shown in [the demo notebook](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/main/docs/demo.ipynb).

## Voronoi-Otsu-Labeling

This algorithm uses [Otsu's thresholding method](https://ieeexplore.ieee.org/document/4310076) in combination with 
[Gaussian blur](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian) and a 
[Voronoi-Tesselation](https://en.wikipedia.org/wiki/Voronoi_diagram) 
approach to label bright objects such as nuclei in an intensity image. The alogrithm has two sigma parameters which allow
you to fine-tune where objects should be cut (`spot_sigma`) and how smooth outlines should be (`outline_sigma`).
This implementation aims to be similar to [Voronoi-Otsu-Labeling in clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb).

![img.png](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/main/docs/voronoi_otsu_labeling.png)

## Seeded Watershed

Starting from an image showing high-intensity membranes and a seed-image where objects have been labeled (e.g. using Voronoi-Otsu-Labeling),
objects are labeled that are constrained by the membranes.

![img.png](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/main/docs/seeded_watershed.png)

## Gaussian blur

Applies a [Gaussian blur](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian) to an
image. This might be useful for denoising, e.g. before applying the Threshold-Otsu method.

![img.png](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/main/docs/gaussian_blur.png)

## Subtract background

Subtracts background using [scikit-image's rolling-ball algorithm](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rolling_ball.html). 
This might be useful, for example to make intensity of membranes more similar in different regions of an image.

![img.png](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/main/docs/subtract_background.png)

## Threshold Otsu

Binarizes an image using [scikit-image's threshold Otsu algorithm](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html), also known as 
[Otsu's method](https://ieeexplore.ieee.org/document/4310076).

![img.png](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/main/docs/threshold_otsu.png)

## Binary watershed

In case objects stick together after thresholding, this implementation of the watershed algorithm might help.
It aims to deliver similar results as [ImageJ's watershed implementation](https://imagej.nih.gov/ij/docs/menus/process.html#watershed).
This function appears to work well in 2D only at the moment.

![img.png](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/main/docs/binary_watershed.png)

## Connected components labeling

Takes a binary image and produces a label image with all separated objects labeled differently. Under the hood, it uses
[scikit-image's label function](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html).

![img.png](https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/raw/main/docs/connected_components_labeling.png)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-segment-blobs-and-things-with-membranes` via [pip]:

    pip install napari-segment-blobs-and-things-with-membranes

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-segment-blobs-and-things-with-membranes" is free and open source software

## Issues

If you encounter any problems, please create a thread on [image.sc] along with a detailed description and tag [@haesleinhuepf].

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

[image.sc]: https://image.sc
[@haesleinhuepf]: https://twitter.com/haesleinhuepf
