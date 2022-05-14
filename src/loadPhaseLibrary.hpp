#ifndef loadPhaseLibrary_hpp
#define loadPhaseLibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadPhaseLibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + phaseConsts
            + calculatePhaseKernel
        ),
        device
    );
}

#endif
