#ifndef loadConsAvgNoMaskLibrary_hpp
#define loadConsAvgNoMaskLibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadConsAvgNoMaskLibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + consAvgNoMaskKernel
        ),
        device
    );
}

#endif
