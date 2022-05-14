#ifndef loadInfAvgLibrary_hpp
#define loadInfAvgLibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadInfAvgLibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + dimConstants
            + infernoConstants
            + dataArraysStruct
            + ConsAvgDataStruct
            + getIndexFuncs
            + calcBurntArea
            + sigmoid
            + calcFlam2
            + calculateWeightedBA
            + infernoConsAvgIg1Flam2Kernel
        ),
        device
    );
}

#endif
