// bro.tensor's safetensors files under the host's access gate
// (embed/embed_access.h; src/api/api.h, "Access"): openSafetensors asks Read
// and saveSafetensors Write about the path, a refusal is an Error carrying the
// gate's sentence, code EACCES and the absolute path, nothing is read or
// written, and with the gate removed everything behaves as before.

#include "api_test_helpers.h"
#include "brotensor/safetensors.h"
#include "embed/embed_access.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace ev = bronze::embed;

std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// The path as the gate is handed it.
std::string gatePath(const fs::path& p) {
    return utf8(fs::absolute(p).lexically_normal());
}

std::string jsStr(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) {
        if (c == '\\' || c == '\'') out += '\\';
        out += c;
    }
    return out + "'";
}

std::string evalString(const std::string& src) {
    auto r = bronze::eval::evalScript(src);
    if (r.thrown) return "<thrown: " + ev::toUtf8(r.value) + ">";
    return ev::isString(r.value) ? ev::toUtf8(r.value) : "<not a string>";
}

// Runs `stmt`; "ok" or the refusal as one comparable string.
std::string syncJs(const std::string& stmt) {
    return evalString("(function() { const T = globalThis.bro.tensor; try { " + stmt +
                      "; return 'ok'; } catch (e) {\n"
                      "  return 'denied|' + e.message + '|' + e.code + '|' + e.path; } })()");
}

std::string refusal(const fs::path& p) {
    const std::string path = gatePath(p);
    return "denied|no files access: " + path + "|EACCES|" + path;
}

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  FAIL: %s\n", what.c_str());
        ++g_fail;
    }
}

} // namespace

int run_api_gate_tests() {
    std::printf("test_js_access_gate...\n");
    g_fail = 0;
    // Per-process: brotensor_test_api and its gcstress twin run at once.
    const fs::path root = fs::temp_directory_path() / ("brotensor_api_gate_" + std::to_string(std::random_device{}()));
    const fs::path ok = root / "ok";
    const fs::path hidden = root / "hidden";
    std::error_code ec;
    fs::create_directories(ok);
    fs::create_directories(hidden);
    {
        namespace st = brotensor::safetensors;
        const float alpha[2] = {1, 2};
        std::vector<st::WriteEntry> entries(1);
        entries[0].name = "alpha";
        entries[0].dtype = st::Dtype::F32;
        entries[0].shape = {1, 2};
        entries[0].host_data = alpha;
        entries[0].bytes = sizeof(alpha);
        st::write_file(utf8(ok / "a.safetensors"), entries);
        st::write_file(utf8(hidden / "a.safetensors"), entries);
    }

    const std::string hiddenPrefix = gatePath(hidden);
    std::atomic<int> calls{0};
    ev::setAccessGate([&](const ev::AccessRequest& q) -> std::string {
        if (q.kind != ev::AccessKind::Read && q.kind != ev::AccessKind::Write) return "";
        ++calls;
        const std::string path(q.subject);
        return path.compare(0, hiddenPrefix.size(), hiddenPrefix) == 0 ? "no files access: " + path : "";
    });

    const std::string tensor = "{x: (() => { const t = T.createTensor(1, 2); t.upload(Float32Array.from([3, 4])); return t; })()}";
    std::string r;
    r = syncJs("T.openSafetensors(" + jsStr(hidden / "a.safetensors") + ")");
    check(r == refusal(hidden / "a.safetensors"), "refused open: " + r);
    r = syncJs("if (T.openSafetensors(" + jsStr(ok / "a.safetensors") + ").count !== 1) throw new Error('count')");
    check(r == "ok", "allowed open: " + r);
    // A refusal is not left behind for the next failure to report.
    r = syncJs("T.openSafetensors(" + jsStr(ok / "missing.safetensors") + ")");
    check(r.rfind("denied|", 0) == 0 && r.find("EACCES") == std::string::npos, "missing file after a refusal: " + r);

    r = syncJs("T.saveSafetensors(" + jsStr(hidden / "out.safetensors") + ", " + tensor + ")");
    check(r == refusal(hidden / "out.safetensors"), "refused save: " + r);
    check(!fs::exists(hidden / "out.safetensors"), "a refused saveSafetensors wrote");
    r = syncJs("T.saveSafetensors(" + jsStr(ok / "out.safetensors") + ", " + tensor + ")");
    check(r == "ok", "allowed save: " + r);
    check(fs::exists(ok / "out.safetensors"), "an allowed saveSafetensors did not write");

    // With the gate removed, nothing asks and everything works as before.
    ev::setAccessGate(nullptr);
    const int before = calls.load();
    r = syncJs("T.openSafetensors(" + jsStr(hidden / "a.safetensors") + ").close()");
    check(r == "ok", "ungated open: " + r);
    r = syncJs("T.saveSafetensors(" + jsStr(hidden / "out.safetensors") + ", " + tensor + ")");
    check(r == "ok", "ungated save: " + r);
    check(fs::exists(hidden / "out.safetensors"), "an ungated saveSafetensors did not write");
    check(calls.load() == before, "a removed gate was still asked");

    fs::remove_all(root, ec);
    return g_fail;
}
