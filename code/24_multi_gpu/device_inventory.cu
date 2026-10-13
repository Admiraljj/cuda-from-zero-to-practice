#include "cuda_helpers.h"
#include <cuda_runtime.h>
#include <iostream>

int main() {
    const int status=book24::require_devices(1);if(status) return status;
    int count=0;if(!book24::check(cudaGetDeviceCount(&count),-1,"device count")) return 1;
    for(int d=0;d<count;++d) {
        cudaDeviceProp properties{};char bus[32]{};
        if(!book24::check(cudaGetDeviceProperties(&properties,d),d,"properties") ||
           !book24::check(cudaDeviceGetPCIBusId(bus,sizeof(bus),d),d,"PCI bus id") ||
           !book24::check(cudaSetDevice(d),d,"select")) return 1;
        int current=-1;if(!book24::check(cudaGetDevice(&current),d,"current device")) return 1;
        std::cout << "local_device=" << d << " current=" << current << " name=" << properties.name
                  << " cc=" << properties.major << '.' << properties.minor << " pci=" << bus
                  << " UVA=" << properties.unifiedAddressing << '\n';
        float* pointer=nullptr;if(!book24::check(cudaMalloc(&pointer,sizeof(float)),d,"allocate")) return 1;
        cudaPointerAttributes attributes{};
        bool ok=book24::check(cudaPointerGetAttributes(&attributes,pointer),d,"pointer attributes");
        if(ok) std::cout << "allocation owner=" << attributes.device << " memory_type=" << attributes.type << '\n';
        ok=book24::check(cudaFree(pointer),d,"free")&&ok;if(!ok) return 1;
    }
    for(int access=0;access<count;++access) {
        for(int owner=0;owner<count;++owner) {
            if(access==owner) continue;
            int can=0;if(!book24::check(cudaDeviceCanAccessPeer(&can,access,owner),access,"peer query")) return 1;
            std::cout << "accessor=" << access << " owner=" << owner << " can_access=" << can << '\n';
        }
    }
    std::cout << "PASS inventory queries; no peer enabled and no kernel launched.\n";return 0;
}
