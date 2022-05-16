#ifndef loadInfAvgScoreLibrary_hpp
#define loadInfAvgScoreLibrary_hpp

#include <stdio.h>
#include <simd/simd.h>
#include <iostream>
#include <fstream>
#include "loadLibraryCommon.hpp"


MTL::Library* loadInfAvgScoreLibrary(MTL::Device* device) {
    return loadLibraryCommon(
        (
            common
            + dimConstants
            + infernoConstants
            + phaseConsts
            + dataArraysStruct
            + ConsAvgObsDataStruct
            + getIndexFuncs
            + calcBurntArea
            + sigmoid
            + calcFlam2
            + calculateWeightedBA
            + infernoConsAvgScoreIg1Flam2Kernel
        ),
        device
    );
}

#endif
