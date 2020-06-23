CXX=g++
LD=ld

OPENCV_PATH=/c/USIT/opencv
OPENCV_VERSIONTAG=244
BOOST_PATH=/c/USIT/boost_1_53_0
BOOST_TARGET=mgw47
BOOST_VERSIONTAG=1_53
MINGW_PATH=/c/MinGW/

# Windows
CXXFLAGS=-O3 -Wall -fmessage-length=0 \
    -I${OPENCV_PATH}/build/include \
    -I${BOOST_PATH}


##
## Use this to utilize the default version of the opencv
## This could lead to an access vioalation error 0xc000007b
## If this happens the dlls of the opencv are faulty, compile them by hand.
##
LINK_OPENCV= \
    ${OPENCV_PATH}/build/x86/mingw/lib/x86/mingw/libopencv_core${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build/x86/mingw/lib/x86/mingw/libopencv_highgui${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build/x86/mingw/lib/x86/mingw/libopencv_imgproc${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build/x86/mingw/lib/x86/mingw/libopencv_objdetect${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build/x86/mingw/lib/x86/mingw/libopencv_contrib${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build/x86/mingw/lib/x86/mingw/libopencv_photo${OPENCV_VERSIONTAG}.dll.a 

## 
## Use this for hand compiled opencv 32bit libs.
## In a mingw shell: 
## $ cd ${OPENCV_PATH}
## $ mkdir build2 ; cd build2
## $ cmake -G "MSYS Makefiles" -DCMAKE_CXX_FLAGS:STRING=-m32 ..
## $ make
##
#LINK_OPENCV= \
    ${OPENCV_PATH}/build2/lib/libopencv_core${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build2/lib/libopencv_highgui${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build2/lib/libopencv_imgproc${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build2/lib/libopencv_objdetect${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build2/lib/libopencv_contrib${OPENCV_VERSIONTAG}.dll.a \
    ${OPENCV_PATH}/build2/lib/libopencv_photo${OPENCV_VERSIONTAG}.dll.a 


LINKFLAGS= ${LINK_OPENCV} \
    -L${MINGW_PATH}/lib  \
    -L${BOOST_PATH}/lib32-msvc-10.0 \
    ${BOOST_PATH}/stage/lib/libboost_filesystem-${BOOST_TARGET}-mt-${BOOST_VERSIONTAG}.a \
    ${BOOST_PATH}/stage/lib/libboost_system-${BOOST_TARGET}-mt-${BOOST_VERSIONTAG}.a \
    ${BOOST_PATH}/stage/lib/libboost_regex-${BOOST_TARGET}-mt-${BOOST_VERSIONTAG}.a
	

maskcmpprf.exe: maskcmpprf.cpp
	$(CXX) -o maskcmpprf.exe $(CXXFLAGS) maskcmpprf.cpp $(LINKFLAGS)

manuseg.exe: manuseg.cpp
	$(CXX) -o manuseg.exe $(CXXFLAGS) manuseg.cpp $(LINKFLAGS)

	
all: maskcmpprf.exe manuseg.exe
