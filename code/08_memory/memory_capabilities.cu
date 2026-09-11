#include <cuda_runtime.h>
#include <iostream>

bool query(int& value, cudaDeviceAttr attribute, const char* name, int device) {
    const cudaError_t status = cudaDeviceGetAttribute(&value, attribute, device);
    if (status != cudaSuccess) {
        std::cerr << name << ": " << cudaGetErrorString(status) << '\n';
        return false;
    }
    std::cout << name << '=' << value << '\n';
    return true;
}

int main() {
    int device = -1;
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        std::cerr << "current device: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    int managed = -1, concurrent = -1, pageable = -1, host_tables = -1;
    std::cout << "device=" << device << '\n';
    if (!query(managed, cudaDevAttrManagedMemory, "managedMemory", device)) return 1;
    if (!query(concurrent, cudaDevAttrConcurrentManagedAccess,
               "concurrentManagedAccess", device)) return 1;
    if (!query(pageable, cudaDevAttrPageableMemoryAccess,
               "pageableMemoryAccess", device)) return 1;
    if (!query(host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
               "pageableMemoryAccessUsesHostPageTables", device)) return 1;

    if (!managed) std::cout << "mode=no managed allocation support\n";
    else if (!concurrent) std::cout << "mode=limited managed support\n";
    else if (!pageable) std::cout << "mode=full support for managed allocations\n";
    else if (!host_tables) std::cout << "mode=system memory, software coherence\n";
    else std::cout << "mode=system memory, hardware coherence\n";
    std::cout << "PASS capability query; no kernel or allocation experiment\n";
    return 0;
}
