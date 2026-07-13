#!/usr/bin/env python3
"""Render the repo's markdown docs into the GitHub Pages site.

The site is three things stapled together:

  index.html          <- README.md
  <name>.html         <- docs/<name>.md
  coverage/index.html <- the gcovr report, dropped in by the workflow

Markdown on github.com renders itself; the reason to publish it here is that
the coverage report has nowhere else to live (it is otherwise a build artifact
that expires), and a report with no docs around it is a dead end. So the docs
come along, and the two link to each other.

Link rewriting is the only fiddly part. The markdown is written to be read in
the repo, so its links point at repo paths — docs/architecture.md, LICENSE,
.github/workflows/codeql.yml. On the site, a link to another *page we render*
must become the .html we rendered; a link to anything else must go back to
GitHub, or it 404s.
"""

import re
import sys
from pathlib import Path

import markdown

REPO = "https://github.com/wlejon/brotensor"
BLOB = f"{REPO}/blob/main"

# Every page we render, in nav order. The key is the output stem.
PAGES = [
    ("index", Path("README.md"), "Home"),
    ("architecture", Path("docs/architecture.md"), "Architecture"),
    ("api", Path("docs/api.md"), "API"),
    ("op-coverage", Path("docs/op-coverage.md"), "Op coverage"),
]

# Rendered stems, for deciding whether a .md link resolves to a page or to GitHub.
STEMS = {stem for stem, _, _ in PAGES}

CSS = """
:root { color-scheme: light dark; --fg:#1f2328; --bg:#fff; --muted:#59636e;
        --line:#d1d9e0; --link:#0969da; --code-bg:#f6f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6edf3; --bg:#0d1117; --muted:#9198a1;
          --line:#3d444d; --link:#4493f8; --code-bg:#151b23; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); line-height:1.6;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
nav { border-bottom:1px solid var(--line); padding:1rem 1.5rem; display:flex;
      flex-wrap:wrap; gap:1.25rem; align-items:baseline; }
nav .brand { font-weight:700; letter-spacing:-.01em; }
nav a { color:var(--link); text-decoration:none; }
nav a:hover { text-decoration:underline; }
nav .spacer { flex:1; }
main { max-width:56rem; margin:0 auto; padding:2rem 1.5rem 5rem; }
main a { color:var(--link); }
h1,h2,h3 { line-height:1.25; margin-top:2rem; }
h1 { letter-spacing:-.02em; }
h2 { border-bottom:1px solid var(--line); padding-bottom:.3rem; }
code { background:var(--code-bg); padding:.15em .35em; border-radius:6px;
       font-size:.9em; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
pre { background:var(--code-bg); padding:1rem; border-radius:8px; overflow-x:auto; }
pre code { background:none; padding:0; }
/* Wide tables scroll inside themselves rather than pushing the page sideways —
   op-coverage.md has some very wide ones. */
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; }
th,td { border:1px solid var(--line); padding:.4rem .7rem; text-align:left; }
th { background:var(--code-bg); }
blockquote { margin:0; padding-left:1rem; border-left:3px solid var(--line); color:var(--muted); }
img { max-width:100%; }
footer { border-top:1px solid var(--line); margin-top:3rem; padding-top:1rem;
         color:var(--muted); font-size:.9em; }
"""

HTML = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
<nav>
  <span class="brand">brotensor</span>
  {nav}
  <span class="spacer"></span>
  <a href="{repo}">GitHub</a>
</nav>
<main>
{body}
<footer>Built from <a href="{blob}">the repository</a> on every push to <code>main</code>.</footer>
</main>
"""


def rewrite_link(href: str) -> str:
    """Repo-relative link -> site link, or back to GitHub if we don't render it."""
    # Leave absolute URLs and in-page anchors alone.
    if re.match(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", href, re.I):
        return href

    m = re.fullmatch(r"(?:\./)?(?:docs/)?([\w.-]+)\.md(#.*)?", href)
    if m and m.group(1) in STEMS:
        return m.group(1) + ".html" + (m.group(2) or "")

    # A real file in the repo that isn't one of our pages (LICENSE, a workflow,
    # a source file). Send it to GitHub rather than leaving a dangling link.
    # Strip a leading "./" as a prefix, not as a character set — lstrip("./")
    # would also eat the dot of ".github/workflows/...".
    return f"{BLOB}/{re.sub(r'^\./', '', href)}"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    out.mkdir(parents=True, exist_ok=True)

    nav = " ".join(f'<a href="{stem}.html">{label}</a>' for stem, _, label in PAGES[1:])
    nav = f'<a href="index.html">Home</a> {nav} <a href="coverage/index.html">Coverage</a>'

    for stem, src, label in PAGES:
        if not src.exists():
            print(f"build_site: missing {src}", file=sys.stderr)
            return 1

        body = markdown.markdown(
            src.read_text(encoding="utf-8"),
            extensions=["fenced_code", "tables", "toc", "sane_lists"],
        )
        body = re.sub(
            r'href="([^"]+)"', lambda m: f'href="{rewrite_link(m.group(1))}"', body
        )
        # Let wide tables scroll on their own instead of widening the page.
        body = body.replace("<table>", '<div class="table-wrap"><table>')
        body = body.replace("</table>", "</table></div>")

        title = "brotensor" if stem == "index" else f"brotensor — {label}"
        (out / f"{stem}.html").write_text(
            HTML.format(title=title, css=CSS, nav=nav, body=body, repo=REPO, blob=BLOB),
            encoding="utf-8",
        )
        print(f"build_site: {src} -> {out / (stem + '.html')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
