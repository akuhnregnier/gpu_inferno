#ifndef loadFlam2Library_hpp
#define loadFlam2Library_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadFlam2Library(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + sigmoid
            + calcFlam2
            + calcFlam2Kernel
        ),
        device
    );
}

#endif
