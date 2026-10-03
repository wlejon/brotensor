// Minimal SPIR-V reflection for the pipeline limit guard. See
// detail/spirv_reflect.h.

#include "detail/spirv_reflect.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace brotensor::detail::vulkan {

namespace {

enum Op : std::uint32_t {
    OpExecutionMode = 16, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22,
    OpTypeVector = 23, OpTypeMatrix = 24, OpTypeArray = 28, OpTypeStruct = 30,
    OpTypePointer = 32, OpConstantTrue = 41, OpConstantFalse = 42, OpConstant = 43,
    OpConstantComposite = 44, OpSpecConstantTrue = 48, OpSpecConstantFalse = 49,
    OpSpecConstant = 50, OpSpecConstantComposite = 51, OpSpecConstantOp = 52,
    OpVariable = 59, OpDecorate = 71, OpExecutionModeId = 331,
};
constexpr std::uint32_t kDecSpecId = 1, kDecBuiltIn = 11, kBuiltInWorkgroupSize = 25;
constexpr std::uint32_t kModeLocalSize = 17, kModeLocalSizeId = 38;
constexpr std::uint32_t kStorageWorkgroup = 4;

struct Type {
    std::uint32_t op = 0;
    std::uint32_t a = 0, b = 0;            // width / component / element, count / length id
    std::vector<std::uint32_t> members;    // struct
};

struct Reflector {
    std::unordered_map<std::uint32_t, Type> types;
    std::unordered_map<std::uint32_t, std::int64_t> values;               // scalar constants
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> composites;
    std::unordered_map<std::uint32_t, std::uint32_t> spec_ids;            // id -> SpecId
    std::uint32_t workgroup_size_id = 0;
    std::string err;

    bool size_of(std::uint32_t id, std::uint64_t* size, std::uint64_t* align, int depth = 0) {
        if (depth > 32) { err = "type nesting too deep"; return false; }
        auto it = types.find(id);
        if (it == types.end()) { err = "unknown type in shared variable"; return false; }
        const Type& t = it->second;
        switch (t.op) {
            case OpTypeBool: *size = 4; *align = 4; return true;
            case OpTypeInt:
            case OpTypeFloat: *size = t.a / 8; *align = t.a / 8; return true;
            case OpTypeVector: {
                std::uint64_t cs = 0, ca = 0;
                if (!size_of(t.a, &cs, &ca, depth + 1)) return false;
                const std::uint64_t n = (t.b == 3) ? 4 : t.b;
                *size = cs * t.b;
                *align = cs * n;
                return true;
            }
            case OpTypeMatrix: {
                std::uint64_t cs = 0, ca = 0;
                if (!size_of(t.a, &cs, &ca, depth + 1)) return false;
                const std::uint64_t stride = (cs + ca - 1) / ca * ca;
                *size = stride * t.b;
                *align = ca;
                return true;
            }
            case OpTypeArray: {
                std::uint64_t es = 0, ea = 0;
                if (!size_of(t.a, &es, &ea, depth + 1)) return false;
                auto v = values.find(t.b);
                if (v == values.end() || v->second < 0) {
                    err = "shared array length is not an evaluable constant";
                    return false;
                }
                const std::uint64_t stride = (es + ea - 1) / ea * ea;
                *size = stride * static_cast<std::uint64_t>(v->second);
                *align = ea;
                return true;
            }
            case OpTypeStruct: {
                std::uint64_t off = 0, max_align = 1;
                for (std::uint32_t m : t.members) {
                    std::uint64_t ms = 0, ma = 0;
                    if (!size_of(m, &ms, &ma, depth + 1)) return false;
                    off = (off + ma - 1) / ma * ma + ms;
                    max_align = std::max(max_align, ma);
                }
                *size = (off + max_align - 1) / max_align * max_align;
                *align = max_align;
                return true;
            }
            default:
                err = "unsupported type in shared variable";
                return false;
        }
    }

    bool eval_spec_op(std::uint32_t opcode, const std::uint32_t* ops, std::uint32_t nops,
                      std::int64_t* out) {
        auto get = [&](std::uint32_t i, std::int64_t* v) {
            if (i >= nops) return false;
            auto it = values.find(ops[i]);
            if (it == values.end()) return false;
            *v = it->second;
            return true;
        };
        std::int64_t x = 0, y = 0;
        if (!get(0, &x)) return false;
        switch (opcode) {
            case 113: case 114: *out = x; return true;   // UConvert / SConvert
            default: break;
        }
        if (!get(1, &y)) return false;
        switch (opcode) {
            case 128: *out = x + y; return true;
            case 130: *out = x - y; return true;
            case 132: *out = x * y; return true;
            case 134: case 135: if (y == 0) return false; *out = x / y; return true;
            case 137: case 138: case 139: if (y == 0) return false; *out = x % y; return true;
            case 194: case 195: *out = x >> y; return true;
            case 196: *out = x << y; return true;
            case 197: *out = x | y; return true;
            case 198: *out = x ^ y; return true;
            case 199: *out = x & y; return true;
            default: return false;
        }
    }
};

}  // namespace

bool reflect_spirv(const std::uint32_t* w, std::size_t n, const std::uint32_t* spec,
                   std::uint32_t nspec, SpirvInfo* out, std::string* why) {
    auto fail = [&](const std::string& m) {
        if (why) *why = m;
        return false;
    };
    if (n < 5 || w[0] != 0x07230203u) return fail("not a SPIR-V module");
    Reflector r;
    SpirvInfo info;
    std::uint32_t local_ids[3] = {0, 0, 0};
    std::vector<std::uint32_t> shared_vars;          // pointer type ids
    std::unordered_map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> ptr_types;  // id -> (storage, pointee)

    for (std::size_t i = 5; i < n;) {
        const std::uint32_t wc = w[i] >> 16;
        const std::uint32_t op = w[i] & 0xffffu;
        if (wc == 0 || i + wc > n) return fail("malformed SPIR-V instruction stream");
        const std::uint32_t* a = w + i + 1;   // operands
        const std::uint32_t na = wc - 1;
        switch (op) {
            case OpDecorate:
                if (na >= 3 && a[1] == kDecSpecId) r.spec_ids[a[0]] = a[2];
                if (na >= 3 && a[1] == kDecBuiltIn && a[2] == kBuiltInWorkgroupSize) r.workgroup_size_id = a[0];
                break;
            case OpExecutionMode:
                if (na >= 5 && a[1] == kModeLocalSize) {
                    info.local[0] = a[2]; info.local[1] = a[3]; info.local[2] = a[4];
                }
                break;
            case OpExecutionModeId:
                if (na >= 5 && a[1] == kModeLocalSizeId) {
                    local_ids[0] = a[2]; local_ids[1] = a[3]; local_ids[2] = a[4];
                }
                break;
            case OpTypeBool: if (na >= 1) r.types[a[0]] = Type{op, 0, 0, {}}; break;
            case OpTypeInt: if (na >= 3) r.types[a[0]] = Type{op, a[1], a[2], {}}; break;  // width, signedness
            case OpTypeFloat: if (na >= 2) r.types[a[0]] = Type{op, a[1], 0, {}}; break;
            case OpTypeVector:
            case OpTypeMatrix:
            case OpTypeArray: if (na >= 3) r.types[a[0]] = Type{op, a[1], a[2], {}}; break;
            case OpTypeStruct:
                if (na >= 1) r.types[a[0]] = Type{op, 0, 0, std::vector<std::uint32_t>(a + 1, a + na)};
                break;
            case OpTypePointer: if (na >= 3) ptr_types[a[0]] = {a[1], a[2]}; break;
            case OpConstantTrue: case OpSpecConstantTrue: if (na >= 2) r.values[a[1]] = 1; break;
            case OpConstantFalse: case OpSpecConstantFalse: if (na >= 2) r.values[a[1]] = 0; break;
            case OpConstant:
            case OpSpecConstant: {
                if (na < 3) break;
                std::int64_t v = static_cast<std::int64_t>(a[2]);
                auto t = r.types.find(a[0]);
                const bool is_int = t != r.types.end() && t->second.op == OpTypeInt;
                if (is_int && t->second.a == 64 && na >= 4) {
                    v = static_cast<std::int64_t>((std::uint64_t(a[3]) << 32) | a[2]);
                } else if (is_int && t->second.b == 0) {
                    v = static_cast<std::int64_t>(a[2]);          // unsigned 32
                } else if (is_int) {
                    v = static_cast<std::int32_t>(a[2]);          // signed 32
                }
                if (op == OpSpecConstant) {
                    auto s = r.spec_ids.find(a[1]);
                    if (s != r.spec_ids.end() && s->second < nspec) v = spec[s->second];
                }
                if (is_int) r.values[a[1]] = v;
                break;
            }
            case OpConstantComposite:
            case OpSpecConstantComposite:
                if (na >= 2) r.composites[a[1]] = std::vector<std::uint32_t>(a + 2, a + na);
                break;
            case OpSpecConstantOp: {
                if (na < 3) break;
                std::int64_t v = 0;
                if (r.eval_spec_op(a[2], a + 3, na - 3, &v)) r.values[a[1]] = v;
                break;
            }
            case OpVariable:
                if (na >= 3 && a[2] == kStorageWorkgroup) shared_vars.push_back(a[0]);
                break;
            default: break;
        }
        i += wc;
    }

    for (int k = 0; k < 3; ++k) {
        if (!local_ids[k]) continue;
        auto v = r.values.find(local_ids[k]);
        if (v == r.values.end()) return fail("LocalSizeId is not an evaluable constant");
        info.local[k] = static_cast<std::uint32_t>(v->second);
    }
    if (r.workgroup_size_id) {
        auto c = r.composites.find(r.workgroup_size_id);
        if (c == r.composites.end() || c->second.size() != 3) return fail("bad WorkgroupSize constant");
        for (int k = 0; k < 3; ++k) {
            auto v = r.values.find(c->second[k]);
            if (v == r.values.end()) return fail("WorkgroupSize component is not evaluable");
            info.local[k] = static_cast<std::uint32_t>(v->second);
        }
    }
    for (std::uint32_t pt : shared_vars) {
        auto p = ptr_types.find(pt);
        if (p == ptr_types.end()) return fail("shared variable without a pointer type");
        std::uint64_t size = 0, align = 0;
        if (!r.size_of(p->second.second, &size, &align)) return fail(r.err);
        info.shared_bytes = (info.shared_bytes + align - 1) / align * align + size;
    }
    *out = info;
    return true;
}

}  // namespace brotensor::detail::vulkan
