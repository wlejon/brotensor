#include "api.h"
#include "api_internal.h"
#include "native_tensor_decl.h"

#include <initializer_list>
#include <string>

extern "C" void bronze_tensor_main(void);

namespace brotensor::api {

namespace {

namespace ev = bronze::embed;

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

bool registerTensorNatives(std::string* error) {
    const char* kTensorCls = "__bro_native.tensor.GpuTensor";

    bool ok =
        ctor(kTensorCls, p(&bro_tensor_GpuTensor_ctor), &bro_tensor_GpuTensor_dtor, bronze::runtime::Finalize::InSweep, {}, error) &&
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

    if (!ok) return false;
    publishPrototype("tensor", kTensorCls, "GpuTensorProto");
    return true;
}

void installTensorJS() {
    ev::runEntry(bronze_tensor_main);
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
