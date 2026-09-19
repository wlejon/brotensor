#include "api.h"
#include "api_internal.h"
#include "native_tensor_decl.h"
#include "native_tensor_batched_decl.h"
#include "native_tensor_attn2_decl.h"
#include "native_tensor_audio_decl.h"
#include "native_tensor_conv_decl.h"
#include "native_tensor_int8_decl.h"
#include "native_tensor_misc_decl.h"

#include <initializer_list>
#include <string>

extern "C" void bronze_tensor_main(void);
extern "C" void bronze_tensor_ext_main(void);
extern "C" void bronze_tensor_batched_main(void);
extern "C" void bronze_tensor_attn2_main(void);
extern "C" void bronze_tensor_audio_main(void);
extern "C" void bronze_tensor_conv_main(void);
extern "C" void bronze_tensor_int8_main(void);
extern "C" void bronze_tensor_misc_main(void);

namespace brotensor::api {

namespace {

namespace ev = bronze::embed;

std::function<std::string(const std::string&)>& pathResolverSlot() {
    static std::function<std::string(const std::string&)> resolver;
    return resolver;
}

ev::NativeSignature sig(const char* ret, std::initializer_list<const char*> params, ev::NativeKind kind) {
    ev::NativeSignature s;
    s.returnType = ret;
    for (const char* p : params) s.paramTypes.emplace_back(p);
    s.kind = kind;
    return s;
}

bool fn(const char* path, void* f, const char* ret, std::initializer_list<const char*> params, std::string* error) {
    return ev::registerNative(path, f, sig(ret, params, ev::NativeKind::Function), error);
}

bool getter(const char* path, void* f, const char* ret, std::string* error) {
    return ev::registerNative(path, f, sig(ret, {}, ev::NativeKind::Getter), error);
}

bool ctor(const char* path, void* f, ev::HandleDestructor dtor, bronze::runtime::Finalize finalize,
          std::initializer_list<const char*> params, std::string* error) {
    ev::NativeSignature s = sig(path, params, ev::NativeKind::Constructor);
    s.className = path;
    s.destructor = dtor;
    s.finalize = finalize;
    return ev::registerNative(path, f, s, error);
}

void publishPrototype(const char* ns, const char* className, const char* protoName) {
    ev::GlobalValue root = ev::globalValue("__bro_native");
    if (!root.found || !ev::isObject(root.value)) return;
    ev::GlobalValue proto = ev::nativeClassPrototype(className);
    if (!proto.found) return;
    ev::Persistent rootSlot(root.value);
    ev::Persistent protoSlot(proto.value);
    ev::Persistent nsSlot(ev::getProperty(rootSlot.get(), ns));
    if (!ev::isObject(nsSlot.get())) return;
    ev::setProperty(nsSlot.get(), protoName, protoSlot.get());
}

template <typename F>
void* p(F* f) { return reinterpret_cast<void*>(f); }

} // namespace

std::string& lastErrorSlot() {
    static thread_local std::string slot;
    return slot;
}

std::string resolvePath(const std::string& path) {
    auto& r = pathResolverSlot();
    return r ? r(path) : path;
}

void setPathResolver(std::function<std::string(const std::string&)> resolver) {
    pathResolverSlot() = std::move(resolver);
}

bool registerTensorNatives(std::string* error) {
    const char* kTensorCls = "__bro_native.tensor.GpuTensor";
    const char* kStCls = "__bro_native.tensor.SafetensorsFile";
    const char* dyn = "dynamic";

    bool ok =
        ctor(kTensorCls, p(&bro_tensor_GpuTensor_ctor), &bro_tensor_GpuTensor_dtor, bronze::runtime::Finalize::InSweep, {}, error) &&
        ctor(kStCls, p(&bro_tensor_SafetensorsFile_ctor), &bro_tensor_SafetensorsFile_dtor, bronze::runtime::Finalize::InSweep, {}, error) &&
        fn("__bro_native.tensor.takeError", p(&bro_tensor_takeError), "str", {}, error) &&
        fn("__bro_native.tensor.GpuTensor_downloadInto", p(&bro_tensor_GpuTensor_downloadInto), "bool", {kTensorCls, "f32[]"}, error) &&
        // safetensors
        fn("__bro_native.tensor.openSafetensors", p(&bro_tensor_openSafetensors), kStCls, {"str"}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_isOpen", p(&bro_tensor_SafetensorsFile_isOpen), "bool", {kStCls}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_count_get", p(&bro_tensor_SafetensorsFile_count_get), "i32", {kStCls}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_nameAt", p(&bro_tensor_SafetensorsFile_nameAt), "str", {kStCls, "i32"}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_dtypeAt", p(&bro_tensor_SafetensorsFile_dtypeAt), "str", {kStCls, "i32"}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_shapeAt", p(&bro_tensor_SafetensorsFile_shapeAt), "f64[]", {kStCls, "i32"}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_nbytesAt", p(&bro_tensor_SafetensorsFile_nbytesAt), "f64", {kStCls, "i32"}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_get", p(&bro_tensor_SafetensorsFile_get), kTensorCls, {kStCls, "str", "i32", "i32", "str"}, error) &&
        fn("__bro_native.tensor.SafetensorsFile_close", p(&bro_tensor_SafetensorsFile_close), "void", {kStCls}, error) &&
        fn("__bro_native.tensor.saveSafetensors", p(&bro_tensor_saveSafetensors), "bool", {"str", dyn, dyn}, error) &&
        // RNG / init
        fn("__bro_native.tensor.randUniform", p(&bro_tensor_randUniform), "void", {dyn, dyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.randn", p(&bro_tensor_randn), "void", {dyn, dyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.randBernoulli", p(&bro_tensor_randBernoulli), "void", {"f64", dyn, dyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.randnTruncated", p(&bro_tensor_randnTruncated), "void", {"f64", "f64", dyn, dyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.xavierInit", p(&bro_tensor_xavierInit), "f64", {kTensorCls, dyn}, error) &&
        // restored dense / loss / concat
        fn("__bro_native.tensor.cast", p(&bro_tensor_cast), "void", {kTensorCls, kTensorCls, "str"}, error) &&
        fn("__bro_native.tensor.softmaxForwardMasked", p(&bro_tensor_softmaxForwardMasked), "void", {kTensorCls, kTensorCls, dyn}, error) &&
        fn("__bro_native.tensor.layernormForward", p(&bro_tensor_layernormForward), "f64[]", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, "f64"}, error) &&
        fn("__bro_native.tensor.maskedMeanPoolForward", p(&bro_tensor_maskedMeanPoolForward), "void", {kTensorCls, dyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.maskedMeanPoolBackward", p(&bro_tensor_maskedMeanPoolBackward), "void", {kTensorCls, dyn, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.softmaxXentFused", p(&bro_tensor_softmaxXentFused), "f64", {kTensorCls, kTensorCls, dyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.softmaxXentFusedBatched", p(&bro_tensor_softmaxXentFusedBatched), "void", {kTensorCls, kTensorCls, dyn, kTensorCls, "i32", kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.concatRows", p(&bro_tensor_concatRows), "void", {dyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.splitRows", p(&bro_tensor_splitRows), "void", {kTensorCls, dyn}, error) &&
        fn("__bro_native.tensor.concatBatchedRows", p(&bro_tensor_concatBatchedRows), "void", {dyn, kTensorCls}, error) &&
        fn("__bro_native.tensor.concatNchwChannels", p(&bro_tensor_concatNchwChannels), "void", {dyn, "i32", "i32", "i32", "i32[]", kTensorCls}, error) &&
        fn("__bro_native.tensor.concatNchwChannelsBackward", p(&bro_tensor_concatNchwChannelsBackward), "void", {kTensorCls, "i32", "i32", "i32", "i32[]", dyn}, error) &&
        // attention family
        fn("__bro_native.tensor.attentionForward", p(&bro_tensor_attentionForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.attentionBackward", p(&bro_tensor_attentionBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.mhaForward", p(&bro_tensor_mhaForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.mhaBackward", p(&bro_tensor_mhaBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.selfAttentionForward", p(&bro_tensor_selfAttentionForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.selfAttentionForwardTrain", p(&bro_tensor_selfAttentionForwardTrain), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.selfAttentionBackward", p(&bro_tensor_selfAttentionBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.selfAttentionBiasForward", p(&bro_tensor_selfAttentionBiasForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.crossAttentionForward", p(&bro_tensor_crossAttentionForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.crossAttentionForwardWithAttn", p(&bro_tensor_crossAttentionForwardWithAttn), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.crossAttentionForwardTrain", p(&bro_tensor_crossAttentionForwardTrain), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.crossAttentionBackward", p(&bro_tensor_crossAttentionBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionForward", p(&bro_tensor_flashAttentionForward), "void", {kTensorCls, kTensorCls, kTensorCls, dyn, "i32", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionWindowedForward", p(&bro_tensor_flashAttentionWindowedForward), "void", {kTensorCls, kTensorCls, kTensorCls, dyn, "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionBackward", p(&bro_tensor_flashAttentionBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, "i32", "bool", kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionQkvoForward", p(&bro_tensor_flashAttentionQkvoForward), "void", {kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, dyn, "i32", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionQkvoBackward", p(&bro_tensor_flashAttentionQkvoBackward), "void", {kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, dyn, "i32", "bool", kTensorCls, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, dyn}, error) &&
        fn("__bro_native.tensor.flashAttentionProjectKv", p(&bro_tensor_flashAttentionProjectKv), "void", {kTensorCls, kTensorCls, dyn, kTensorCls, dyn, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionQWithKvCachedForward", p(&bro_tensor_flashAttentionQWithKvCachedForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, kTensorCls, dyn, dyn, "i32", "bool", kTensorCls}, error) &&
        fn("__bro_native.tensor.resblockForward", p(&bro_tensor_resblockForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, dyn, "i32", "i32", "i32", "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.resblockBackward", p(&bro_tensor_resblockBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, dyn, "i32", "i32", "i32", "i32", "i32", "i32", "f64", kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, kTensorCls, kTensorCls, kTensorCls, dyn, dyn, dyn}, error) &&
        fn("__bro_native.tensor.createTensor", p(&bro_tensor_createTensor), kTensorCls, {"i32", "i32", "str"}, error) &&
        fn("__bro_native.tensor.GpuTensor_rows_get", p(&bro_tensor_GpuTensor_rows_get), "i32", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_cols_get", p(&bro_tensor_GpuTensor_cols_get), "i32", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_size_get", p(&bro_tensor_GpuTensor_size_get), "i32", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_bytes_get", p(&bro_tensor_GpuTensor_bytes_get), "i32", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_zero", p(&bro_tensor_GpuTensor_zero), "void", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_resize", p(&bro_tensor_GpuTensor_resize), "void", {kTensorCls, "i32", "i32", "str"}, error) &&
        fn("__bro_native.tensor.GpuTensor_dtype", p(&bro_tensor_GpuTensor_dtype), "str", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_clone", p(&bro_tensor_GpuTensor_clone), kTensorCls, {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_upload", p(&bro_tensor_GpuTensor_upload), "void", {kTensorCls, "f32[]"}, error) &&
        fn("__bro_native.tensor.GpuTensor_download", p(&bro_tensor_GpuTensor_download), "f32[]", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_uploadFp16", p(&bro_tensor_GpuTensor_uploadFp16), "void", {kTensorCls, "u16[]"}, error) &&
        fn("__bro_native.tensor.GpuTensor_downloadFp16", p(&bro_tensor_GpuTensor_downloadFp16), "u16[]", {kTensorCls}, error) &&
        fn("__bro_native.tensor.GpuTensor_uploadInt8", p(&bro_tensor_GpuTensor_uploadInt8), "void", {kTensorCls, "i8[]"}, error) &&
        fn("__bro_native.tensor.GpuTensor_downloadInt8", p(&bro_tensor_GpuTensor_downloadInt8), "i8[]", {kTensorCls}, error) &&
        getter("__bro_native.tensor.available", p(&bro_tensor_available_get), "bool", error) &&
        getter("__bro_native.tensor.backend", p(&bro_tensor_backend_get), "str", error) &&
        fn("__bro_native.tensor.init", p(&bro_tensor_init), "void", {}, error) &&
        fn("__bro_native.tensor.sync", p(&bro_tensor_sync), "void", {}, error) &&
        fn("__bro_native.tensor.linearForward", p(&bro_tensor_linearForward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.linearBackward", p(&bro_tensor_linearBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.reluForward", p(&bro_tensor_reluForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.reluBackward", p(&bro_tensor_reluBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.tanhForward", p(&bro_tensor_tanhForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.tanhBackward", p(&bro_tensor_tanhBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.sigmoidForward", p(&bro_tensor_sigmoidForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.sigmoidBackward", p(&bro_tensor_sigmoidBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.addInplace", p(&bro_tensor_addInplace), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.addScalarInplace", p(&bro_tensor_addScalarInplace), "void", {kTensorCls, "f64"}, error) &&
        fn("__bro_native.tensor.scaleInplace", p(&bro_tensor_scaleInplace), "void", {kTensorCls, "f64"}, error) &&
        fn("__bro_native.tensor.mulInplace", p(&bro_tensor_mulInplace), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.clamp", p(&bro_tensor_clamp), "void", {kTensorCls, "f64", "f64"}, error) &&
        fn("__bro_native.tensor.siluForward", p(&bro_tensor_siluForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.siluBackward", p(&bro_tensor_siluBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.geluForward", p(&bro_tensor_geluForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.geluBackward", p(&bro_tensor_geluBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.geluExactForward", p(&bro_tensor_geluExactForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.geluExactBackward", p(&bro_tensor_geluExactBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.quickGeluForward", p(&bro_tensor_quickGeluForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.quickGeluBackward", p(&bro_tensor_quickGeluBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.swigluForward", p(&bro_tensor_swigluForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.swigluBackward", p(&bro_tensor_swigluBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.gegluForward", p(&bro_tensor_gegluForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.gegluBackward", p(&bro_tensor_gegluBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.gegluExactForward", p(&bro_tensor_gegluExactForward), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.gegluExactBackward", p(&bro_tensor_gegluExactBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.softmaxForward", p(&bro_tensor_softmaxForward), "void", {kTensorCls, kTensorCls, "f64"}, error) &&
        fn("__bro_native.tensor.softmaxBackward", p(&bro_tensor_softmaxBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.matmul", p(&bro_tensor_matmul), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.matmulBackward", p(&bro_tensor_matmulBackward), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.conv2dForward", p(&bro_tensor_conv2dForward), "void", {kTensorCls, kTensorCls, "dynamic", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv2dBackwardInput", p(&bro_tensor_conv2dBackwardInput), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv2dBackwardWeight", p(&bro_tensor_conv2dBackwardWeight), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.conv2dBackwardBias", p(&bro_tensor_conv2dBackwardBias), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.upsampleNearest2xForward", p(&bro_tensor_upsampleNearest2xForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.upsampleNearest2xBackward", p(&bro_tensor_upsampleNearest2xBackward), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.upsampleBilinear2xForward", p(&bro_tensor_upsampleBilinear2xForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.upsampleBilinear2xBackward", p(&bro_tensor_upsampleBilinear2xBackward), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.downsampleAvg2xForward", p(&bro_tensor_downsampleAvg2xForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.downsampleAvg2xBackward", p(&bro_tensor_downsampleAvg2xBackward), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.sumRows", p(&bro_tensor_sumRows), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.sumCols", p(&bro_tensor_sumCols), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.argmaxRows", p(&bro_tensor_argmaxRows), "void", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.copyD2D", p(&bro_tensor_copyD2D), "void", {kTensorCls, "i32", kTensorCls, "i32", "i32"}, error) &&
        fn("__bro_native.tensor.nchwToSequence", p(&bro_tensor_nchwToSequence), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_tensor_sequenceToNchw", p(&bro_tensor_sequenceToNchw), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.sequenceToNchw", p(&bro_tensor_sequenceToNchw), "void", {kTensorCls, "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.interp2dForward", p(&bro_tensor_interp2dForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.interp2dAlignCornersForward", p(&bro_tensor_interp2dAlignCornersForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.unfold2dForward", p(&bro_tensor_unfold2dForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.l2NormalizeNchwForward", p(&bro_tensor_l2NormalizeNchwForward), "void", {kTensorCls, "i32", "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.convexUpsampleForward", p(&bro_tensor_convexUpsampleForward), "void", {kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.mseVecForward", p(&bro_tensor_mseVecForward), "f64", {kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.mseVecBackward", p(&bro_tensor_mseVecBackward), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.mseVecPerSample", p(&bro_tensor_mseVecPerSample), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.layernormBackward", p(&bro_tensor_layernormBackward), "void", {kTensorCls, kTensorCls, kTensorCls, "f64", kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.layernormForwardInferenceBatched", p(&bro_tensor_layernormForwardInferenceBatched), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, "f64"}, error) &&
        fn("__bro_native.tensor.layernormForwardInferenceBatchedFp16", p(&bro_tensor_layernormForwardInferenceBatchedFp16), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, "f64"}, error) &&
        fn("__bro_native.tensor.rmsNormForward", p(&bro_tensor_rmsNormForward), "void", {kTensorCls, kTensorCls, "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.rmsNormBackward", p(&bro_tensor_rmsNormBackward), "void", {kTensorCls, kTensorCls, kTensorCls, "f64", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.groupNormForward", p(&bro_tensor_groupNormForward), "void", {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.groupNormBackward", p(&bro_tensor_groupNormBackward), "void", {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", "i32", "i32", "i32", "f64", kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.ropeForward", p(&bro_tensor_ropeForward), "void", {kTensorCls, "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.ropeBackward", p(&bro_tensor_ropeBackward), "void", {kTensorCls, "i32", "i32", "i32", "f64", kTensorCls}, error) &&
        fn("__bro_native.tensor.ropeApply", p(&bro_tensor_ropeApply), "void", {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.ropeApplyBackward", p(&bro_tensor_ropeApplyBackward), "void", {kTensorCls, "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.modulate", p(&bro_tensor_modulate), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.broadcastMul", p(&bro_tensor_broadcastMul), "void", {kTensorCls, kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.attentionTokenMoments", p(&bro_tensor_attentionTokenMoments), "void", {kTensorCls, "i32", "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.buildSlotMask", p(&bro_tensor_buildSlotMask), "void", {kTensorCls, "i32", "i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.buildCausalMaskRow", p(&bro_tensor_buildCausalMaskRow), "void", {"i32", "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.flashAttentionDecode", p(&bro_tensor_flashAttentionDecode), "void", {kTensorCls, kTensorCls, kTensorCls, "i32", "i32", kTensorCls, "bool", "i32", "f64", "i32"}, error) &&
        fn("__bro_native.tensor.flashAttentionDecodeMasked", p(&bro_tensor_flashAttentionDecodeMasked), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, "i32", kTensorCls, "bool", "i32", "f64", "i32"}, error) &&
        fn("__bro_native.tensor.kvCacheAppend", p(&bro_tensor_kvCacheAppend), "void", {kTensorCls, kTensorCls, "i32", kTensorCls, kTensorCls}, error) &&
        fn("__bro_native.tensor.embeddingLookupForward", p(&bro_tensor_embeddingLookupForward), "void", {kTensorCls, kTensorCls, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.embeddingLookupBackward", p(&bro_tensor_embeddingLookupBackward), "void", {kTensorCls, kTensorCls, "i32", kTensorCls}, error) &&
        fn("__bro_native.tensor.sgdStep", p(&bro_tensor_sgdStep), "void", {kTensorCls, kTensorCls, kTensorCls, "f64", "f64"}, error) &&
        fn("__bro_native.tensor.adamStep", p(&bro_tensor_adamStep), "void", {kTensorCls, kTensorCls, kTensorCls, kTensorCls, "f64", "f64", "f64", "f64", "i32"}, error);

    // The restored free-function groups: each native file owns its own table
    // so the files stay readable (see native_register.h).
    ok = ok &&
        registerTensorNatives_batched(error) &&
        registerTensorNatives_attn2(error) &&
        registerTensorNatives_audio(error) &&
        registerTensorNatives_conv(error) &&
        registerTensorNatives_int8(error) &&
        registerTensorNatives_misc(error);

    if (!ok) return false;
    publishPrototype("tensor", kTensorCls, "GpuTensorProto");
    publishPrototype("tensor", kStCls, "SafetensorsFileProto");
    return true;
}

void installTensorJS() {
    // js/tensor.js first (it mounts bro.tensor and GpuTensor), then
    // js/tensor_ext.js, which decorates that namespace with the attention /
    // loss / concat family and reads GpuTensor for its argument checks.
    ev::runEntry(bronze_tensor_main);
    ev::runEntry(bronze_tensor_ext_main);
    // The restored groups decorate the same namespace and read GpuTensor for
    // their argument checks, so they all run after those two.
    ev::runEntry(bronze_tensor_batched_main);
    ev::runEntry(bronze_tensor_attn2_main);
    ev::runEntry(bronze_tensor_audio_main);
    ev::runEntry(bronze_tensor_conv_main);
    ev::runEntry(bronze_tensor_int8_main);
    ev::runEntry(bronze_tensor_misc_main);
}

void installTensor() {
    auto g = ev::globalValue("__bro_native");
    if (!g.found || !ev::isObject(g.value)) {
        ev::Value obj = ev::createObject();
        ev::registerGlobal("__bro_native", obj);
    }
    auto bn = ev::globalValue("__bro_native").value;
    ev::Persistent bnSlot(bn);
    ev::Persistent tensorSlot(ev::createObject());
    ev::setProperty(bnSlot.get(), "tensor", tensorSlot.get());

    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        ev::setProperty(gt.value, "__bro_native", bnSlot.get());
    }

    registerTensorNatives();
    installTensorJS();
}

brotensor::Tensor* getTensorFromHandle(bronze::Value val) {
    void* data = bronze::embed::handleData(val);
    if (!data) return nullptr;
    return toTensor(data);
}

bronze::Value createGpuTensorValue(brotensor::Tensor tensor) {
    auto* h = new GpuTensorHandle();
    h->tensor = std::move(tensor);
    return bronze::embed::wrapNative(h, "__bro_native.tensor.GpuTensor");
}

} // namespace brotensor::api
