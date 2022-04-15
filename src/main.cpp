#include "inferno.hpp"

int main() {
    use_metal();
    return 0;
}

// Failed library load attempts:

// These don't seem to work
// NS::String* filePath = NS::String::alloc()->string("/Users/alexander/Library/Developer/Xcode/DerivedData/gpu_inferno-aldgfitbljzeowchzwmclnlcxswy/Build/Products/Debug/default.metallib", NS::StringEncoding::ASCIIStringEncoding);
// MTL::Library* pLibrary = _pDevice->newLibrary(filePath, NULL);
// MTL::Library* pLibrary = _pDevice->newLibrary( NS::String::string("/Users/alexander/Documents/PhD/gpu_inferno/gpu_inferno/compute.metal", NS::UTF8StringEncoding), nullptr, &pError );

// Doesn't work in standalone compiled binary (but works when running within Xcode itself), why?
// MTL::Library* pLibrary = _pDevice->newDefaultLibrary();

// Error checking now part of loadLibrary func.
//    if ( !pLibrary )
//    {
//        __builtin_printf( "%s", pError->localizedDescription()->utf8String() );
//        assert( false );
//    }
