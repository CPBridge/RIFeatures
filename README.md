# Rotation Invariant Features

A small C++ library for calculating rotation invariant image features.

![A Rotation Invariant Basis Function Set](README.png?raw=true)

### Objectives

The purpose of this library is to calculate rotation invariant features from 2D
images. These are a set of features that describe circular image patches in an
image in a way that is invariant to the orientation of the patch. They can
therefore be used to detect objects in any orientation in a straightforward way.

The emphasis is on providing a highly efficient implementation that:
* Integrates well with random forests machine learning algorithms.
* Only calculates the required features 'on-the-fly'.
* Uses Fourier-domain calculations to improve calculation speeds.
* Is thread-safe to allow learning algorithms using OpenMP multi-threading to use
extract features avoiding data-races (note that only the functions that are explicitly
described as thread-safe in the documentation should be used within parallel segments).

A detailed description of how the feature extraction method works, an evalution,
and citations of relevant scientific literature can be found in my
[DPhil (PhD) thesis](https://chrisbridge.science/docs/bridge_thesis.pdf).
You can find out more about how I am using the library on my
[website](https://chrisbridge.science/).

**Disclaimer** This is research-quality code I have provided to enable others
to understand, reproduce and develop my research. It is **not** intended to be
production quality code. It expects to be used in a certain way and there are
many ways to break it by not using it in this way resulting in runtime crashes
or unexpected results. There is very minimal error checking to detect these cases.
Follow the examples and documentation closely though and it should be easy to use
it correctly.

### Dependencies

* A C++ compiler supporting the C++11 standard (requires a relatively modern version of your compiler).
* A C++ compiler supporting the [OpenMP](http://openmp.org/wp/) standard (includes most major compilers on major platforms including MSVC, g++ and clang).
* The [OpenCV](http://opencv.org) library. Tested on version 4.2 but most fairly recent
versions should be compatible. If you are using GNU/Linux, there will probably
be a suitable packaged version in your distribution's repository.
* The [boost](http://www.boost.org) [special functions](http://www.boost.org/doc/libs/1_62_0/libs/math/doc/html/special.html) library. Again there is likely to be a suitable packaged version on your GNU/Linux distribution.

### Installation and Compilation

The library consists of a shared object library (`librifeatures.so` or `.dll` on
Windows) and a set of headers.

In order to use the library you will need to compile all the source files in the
`src` directory to create the object file and install it somewhere on the dynamic linker
path on your system. You will also need to copy the header files (in the `include`
directory) to somewhere your compiler will find them, or point your compiler at them each time.

A makefile is provided to automate these tasks on most GNU/Linux operating systems
using the `g++` compiler. To use this, navigate to the `build` directory and
execute the following commands:

```bash
$ make
$ sudo make install
```

You can also compile and install a debug version of the library (`librifeaturesd.so`) with:

```bash
$ make debug
$ sudo make install-debug
```

You can uninstall both the release and debug versions with:

```bash
$ sudo make uninstall
```

For other environments you will need to work out how to do these steps yourself.

Once installed, to use the library in your C++ code, you just need to include the
following header file in your source files:
```c++
#include <RIFeatures/RIFeatExtractor.hpp>
```

Then when compiling your code, make sure to link against OpenCV and the `librifeatures.so`
shared object, as well as using c++11 or later and the OpenMP extensions. E.g. to compile a source file called `my_program.cpp` to produce an executable called `my_program`, use something like the following:

```bash
$ g++ -std=c++11 -fopenmp -o my_program my_program.cpp `pkg-config --libs opencv4` -lrifeatures
```

If you have decided not to install the library on your system as above, you just
need to add the `include` directory to the list of directories your compiler looks at
for headers, and add the compiled shared object file to the linker dependencies.

### Example

The repository includes an example file (`example/rotinv_test.cpp`) with plenty of comments that demonstrates how to use the basic functionality of the library.

You can compile it (once the library has been installed, as above) by running the
following from the `build` directory:

```bash
$ make example
```

You can then run it with on a video file (`path/to/some/video/file.avi`):

```bash
$ ./example path/to/some/video/file.avi
```

### Documentation

The full documentation can be found [here](https://cpbridge.github.io/RIFeatures/) or generated from the source using Doxygen. To do this, install the [Doxygen](http://doxygen.org)
tool (available in most GNU/Linux repositories) and from the `doc` directory run.

```bash
$ doygen Doxyfile
```

This will generate an html version of the documentation in the `html` subdirectory,
which you can view by opening `index.html` in your browser.

### Publications

This library was used in the following publications:
* C.P. Bridge, “Computer-Aided Analysis of Fetal Cardiac Ultrasound Videos”, DPhil Thesis, University of Oxford, 2017. Available on [my website](https://chrisbridge.science/publications.html).
* C.P. Bridge, C. Ioannou, and J.A. Noble, “Automated Annotation and Quantitative Description of Ultrasound Videos of the Fetal Heart”, *Medical Image Analysis* 36 (Feb. 2017), pp. 147-161
* C.P. Bridge, Christos Ioannou, and J.A. Noble, “Localizing Cardiac Structures in Fetal Heart Ultrasound Video”, *Machine Learning in Medical Imaging Workshop, MICCAI, 2017*, pp. 246-255. Original article available [here](https://link.springer.com/chapter/10.1007/978-3-319-67389-9_29). Authors' manuscript available on [my website](https://chrisbridge.science/publications.html).

If you use this library in your research, please consider citing these papers. Other relevant publications include:

* C.P. Bridge and J.A. Noble, “Object Localisation In Fetal Ultrasound Images Using Invariant Features”. Proceedings of the IEEE International Symposium on Biomedical Imaging, New York City, 2015

Finally, much of this work is based heavily on the following paper, with which
I have no association:

* K. Liu et al. “Rotation-Invariant HOG Descriptors Using Fourier Analysis in Polar
and Spherical Coordinates”. In: International Journal of Computer Vision 106.3
(2014), pp. 342–364

### Author

Written by Christopher Bridge at the University of Oxford's Institute of Biomedical
Engineering.

The repository also includes some code written by J-P Moreau for calculation of the
struve function (the two functions in `struve.cpp`) that was put up with no licensing
information [here](http://jean-pierre.moreau.pagesperso-orange.fr/c_function2.html). For convenience I have included a slightly modified versions of these files. If you feel
that I have misappropriated your code, please get in touch.

### Licence

This library is available under the GNU General Public License (GPL v3). For more information, see the licence file.
