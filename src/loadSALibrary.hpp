#ifndef loadSALibrary_hpp
#define loadSALibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadSALibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + dimConstants
            + infernoConstants
            + sigmoid
            + calcFlam2
            + calcBurntArea
            + getIndexFuncs
            + SAStruct
            + SAInfernoIg1Flam2Kernel
        ),
        device
    );
}

#endif
