#include <cstdio>
#include <stdexcept>

class Trace {
public:
    explicit Trace(const char* name) : name_(name) {
        std::printf("construct %s\n", name_);
    }
    ~Trace() noexcept {
        std::printf("destroy %s\n", name_);
    }
    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;
private:
    const char* name_;
};

void work(bool fail) {
    Trace outer("outer");
    {
        Trace inner("inner");
        if (fail) throw std::runtime_error("planned failure");
    }
    std::puts("normal work completed");
}

int main() {
    std::puts("NORMAL");
    work(false);
    std::puts("EXCEPTION");
    try {
        work(true);
    } catch (const std::runtime_error& error) {
        std::printf("caught: %s\n", error.what());
    }
    std::puts("PASS scope trace");
    return 0;
}
