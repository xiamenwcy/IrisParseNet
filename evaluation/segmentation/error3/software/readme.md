IRISSEG Software
================


Dependencies
------------

 * [OpenCV](http://opencv.org/) version 2.44
 * [Boost](http://www.boost.org/) version 1.53

Description
-----------

This software package contains two programs:

### manuseg

This is the segmentation program which uses the elliptical and polynomial data and an iris image to extract a mask.

### maskcmpprf

This is the comparison program for masks, it takes to masks and outputs the precision, recall and F1-measure of the masks.

Note
----

The package contains a makefiles for linux and windows. The windows make files uses the [MinGW](http://mingw.org/) environment which is a prerequisite for building the software on windows with the contained makefile. The package also contains pre-compiled windows binaries (32-bit) with required dlls and should be usable in windows without any further dependencies.
