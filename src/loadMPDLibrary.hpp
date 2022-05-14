#ifndef loadMPDLibrary_hpp
#define loadMPDLibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadMPDLibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + phaseConsts
            + calculateMPDKernel
        ),
        device
    );
}

#endif
