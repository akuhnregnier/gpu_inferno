#ifndef loadLibrary_hpp
#define loadLibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadLibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + dimConstants
            + infernoConstants
            + qsatWat
            + calc_c_comps_triffid_leaf
            + calcIgnitions
            + sigmoid
            + calcFlamGeneral
            + calcFlam2
            + calcBurntArea
            + getIndexFuncs
            + setGetElementFuncs
            + dataArraysStruct
            + multiTimestepInfernoGeneralKernel
            + multiTimestepInfernoIg1Flam2Kernel
        ),
        device
    );
}

#endif
