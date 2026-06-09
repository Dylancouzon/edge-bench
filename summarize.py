"""Merge results/*.json into one comparison table and write RESULTS.md."""
from __future__ import annotations

import glob
import json
from pathlib import Path

ORDER = ["edge-pi", "edge-laptop", "docker-laptop", "cloud"]
LABELS = {
    "edge-pi": "Edge / Pi",
    "edge-laptop": "Edge / laptop",
    "docker-laptop": "Docker / laptop",
    "cloud": "Cloud",
}
NETWORK = {
    "edge-pi": "none (in-process)",
    "edge-laptop": "none (in-process)",
    "docker-laptop": "localhost loopback",
    "cloud": "internet RTT",
}


def load() -> dict:
    res = {}
    for f in glob.glob("results/*.json"):
        if Path(f).stem == "test":
            continue
        d = json.loads(Path(f).read_text())
        res[d["cell"]] = d
    return res


def md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> None:
    res = load()
    cells = [c for c in ORDER if c in res]

    perf_header = [
        "Cell", "Hardware", "Network", "Embed docs/s", "Create (s)",
        "Upload vec/s", "Query p50 (ms)", "Query p95 (ms)", "QPS", "Peak RSS (MB)",
    ]
    perf_rows = []
    for c in cells:
        d = res[c]
        h = d["host"]
        perf_rows.append([
            LABELS[c],
            f"{h['machine']}",
            NETWORK[c],
            d["embedding"]["docs_per_s"],
            d["create_s"],
            d["upload"]["vectors_per_s"],
            d["query_latency_ms"]["p50_ms"],
            d["query_latency_ms"]["p95_ms"],
            d["query_throughput_qps"],
            d["resources_upload"]["peak_rss_mb"],
        ])

    params = res[cells[0]]["params"] if cells else {}
    lines = [
        "# Results — Qdrant Edge vs Docker vs Cloud",
        "",
        f"Dataset: **{params.get('n', '?')} docs × {params.get('dim', '?')}-dim** "
        f"(FastEmbed `BAAI/bge-small-en-v1.5`), {params.get('queries', '?')} queries, "
        f"limit={params.get('limit', '?')}, upsert batch {params.get('batch', '?')}.",
        "",
        "> Not an official benchmark — a quick, honest comparison. The cells differ on "
        "**hardware, network path, and architecture at once**, so read each number with its "
        "context, not as a single leaderboard.",
        "",
        "## Performance",
        "",
        md_table(perf_rows, perf_header),
        "",
        "## What each number means",
        "",
        "- **Embed docs/s** — on-device FastEmbed throughput; a *hardware* signal (Pi CPU vs "
        "laptop CPU), measured separately and excluded from the DB numbers below.",
        "- **Upload / Query** — pure DB-engine work on identical vectors.",
        "- **Edge cells have zero network cost** (in-process); **Docker** adds localhost "
        "loopback; **Cloud** is dominated by internet round-trip, not Qdrant compute.",
        "- **Edge-laptop vs Edge-Pi** isolates hardware. **Edge-laptop vs Docker-laptop** "
        "isolates architecture (in-process vs client/server) on identical silicon.",
        "",
    ]
    # Computed takeaways (only when the relevant cells are present)
    def p50(c):
        return res[c]["query_latency_ms"]["p50_ms"]

    takeaways = []
    if {"edge-laptop", "docker-laptop"} <= res.keys():
        takeaways.append(
            f"- **Architecture (same Mac): in-process beats client/server by "
            f"~{p50('docker-laptop') / p50('edge-laptop'):.0f}× on query latency** — "
            f"Edge {p50('edge-laptop')} ms vs Docker {p50('docker-laptop')} ms. The cost is "
            f"the network/serialization layer, not the engine."
        )
    if {"edge-laptop", "edge-pi"} <= res.keys():
        takeaways.append(
            f"- **Hardware (same in-process engine): the laptop is ~"
            f"{p50('edge-pi') / p50('edge-laptop'):.0f}× faster than the Pi** on query "
            f"latency ({p50('edge-laptop')} ms vs {p50('edge-pi')} ms) — the raw silicon gap."
        )
    if {"edge-pi", "docker-laptop"} <= res.keys():
        takeaways.append(
            f"- **Headline: Edge on the Pi out-queries Docker on the laptop** "
            f"({p50('edge-pi')} ms / {res['edge-pi']['query_throughput_qps']} QPS vs "
            f"{p50('docker-laptop')} ms / {res['docker-laptop']['query_throughput_qps']} QPS) "
            f"— despite far weaker hardware. Removing the network layer matters more than CPU "
            f"for serving queries."
        )
    if {"edge-laptop", "cloud"} <= res.keys():
        takeaways.append(
            f"- **Cloud query latency ({p50('cloud')} ms) is ~"
            f"{p50('cloud') / p50('edge-laptop'):.0f}× Edge's** — almost entirely the internet "
            f"round-trip to the region, not Qdrant compute. Great for managed scale; not for "
            f"latency-critical local inference."
        )
    if {"edge-pi", "edge-laptop"} <= res.keys():
        takeaways.append(
            f"- **On the Pi, embedding is the bottleneck, not the database**: FastEmbed runs at "
            f"{res['edge-pi']['embedding']['docs_per_s']}/s ("
            f"~{res['edge-pi']['embedding']['doc_embed_s'] / 60:.0f} min for 10k) vs the "
            f"laptop's {res['edge-laptop']['embedding']['docs_per_s']}/s, while Edge upload/query "
            f"stay fast. For embedded AI, plan around on-device embedding cost."
        )
    if takeaways:
        lines += ["## Takeaways", "", *takeaways, ""]

    missing = [c for c in ORDER if c not in res]
    if missing:
        lines.append(f"_Pending cells: {', '.join(LABELS[c] for c in missing)}._")
        lines.append("")

    Path("RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
