#ifndef loadConsAvgLibrary_hpp
#define loadConsAvgLibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadConsAvgLibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + consAvgKernel
        ),
        device
    );
}

#endif
