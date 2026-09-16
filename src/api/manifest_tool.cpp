#include "api.h"
#include "embed/embed.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: brotensor-native-manifest <out.json>\n");
        return 2;
    }
    std::string err;
    if (!brotensor::api::registerTensorNatives(&err)) {
        std::fprintf(stderr, "brotensor-native-manifest: %s\n", err.c_str());
        return 1;
    }
    if (!bronze::embed::writeNativeManifest(argv[1], &err)) {
        std::fprintf(stderr, "brotensor-native-manifest: %s\n", err.c_str());
        return 1;
    }
    return 0;
}
