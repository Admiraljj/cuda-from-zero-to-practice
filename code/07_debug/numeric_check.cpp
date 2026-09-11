#include <cmath>
#include <iostream>
#include <limits>

bool close(float got, float expected, float atol, float rtol) {
    if (!std::isfinite(got) || !std::isfinite(expected)
        || !std::isfinite(atol) || !std::isfinite(rtol)
        || atol < 0 || rtol < 0) return false;
    const double difference = std::fabs(static_cast<double>(got) - expected);
    const double allowed = static_cast<double>(atol)
                         + static_cast<double>(rtol) * std::fabs(static_cast<double>(expected));
    return difference <= allowed;
}

int main() {
    struct Case {
        const char* name;
        float got, expected, atol, rtol;
        bool wanted;
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const Case cases[] = {
        {"exact", 2.0f, 2.0f, 0.0f, 0.0f, true},
        {"near_zero", 1e-7f, 0.0f, 1e-6f, 1e-5f, true},
        {"relative", 100001.0f, 100000.0f, 1e-6f, 2e-5f, true},
        {"too_far", 2.0f, 1.0f, 1e-6f, 1e-5f, false},
        {"nan", nan, 1.0f, 1e-6f, 1e-5f, false},
        {"infinity", inf, inf, 1e-6f, 1e-5f, false},
        {"negative_tolerance", 1.0f, 1.0f, -1.0f, 0.0f, false},
        {"invalid_tolerance", 1.0f, 1.0f, inf, 0.0f, false},
    };
    bool ok = true;
    for (const Case& test : cases) {
        const bool result = close(test.got, test.expected, test.atol, test.rtol);
        std::cout << test.name << ": accepted=" << result << '\n';
        if (result != test.wanted) ok = false;
    }
    if (!ok) return 1;
    std::cout << "PASS numeric checker: 8 CPU cases\n";
    return 0;
}
