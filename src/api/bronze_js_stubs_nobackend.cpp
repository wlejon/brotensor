// No-op entry stubs for brotensor's Bronze-compiled JS. Linked in place of the
// compiled objects on a target brass has no code generator for
// (BRASS_HOST_BACKEND OFF; x86_64 and AArch64, Apple Silicon included, compile
// the JS natively), and always into brotensor-native-manifest, which has to
// link before any JS is compiled.
extern "C" {
void bronze_tensor_main() {}
void bronze_tensor_ext_main() {}
void bronze_tensor_batched_main() {}
void bronze_tensor_attn2_main() {}
void bronze_tensor_audio_main() {}
void bronze_tensor_conv_main() {}
void bronze_tensor_int8_main() {}
void bronze_tensor_misc_main() {}
void bronze_tensor_extra_main() {}
}
