# Owner-proposed use cases: research and verdicts

Date: 2026-07-31. These two candidates came from the project owner and were
evaluated with the same rigor as machine-generated candidates: find the
strongest honest version of each, then find the strongest objection. All
project facts below (bandwidth-bound decoding, 31.4 MB/s measured PSRAM,
2.98 tok/s / 8.3 MB per token on the 15M model, 256-token context, no
batched prefill) are taken from `docs/STATUS.md` and `docs/PERFORMANCE.md`
and treated as ground truth, not re-derived.

---

## Candidate A — LoRa mesh companion

### 1. Hardware reality

Both mainstream LoRa transceiver families are plain SPI peripherals and pose
no interfacing problem for a Teensy 4.1:

- **RFM95W** (SX1276-derived): SPI, 3.3 V logic, **sleep 0.2–1 µA, standby
  1.6–1.8 mA, RX 10.8–12.1 mA, TX 20 mA@+7 dBm / 87 mA@+17 dBm / 120 mA@+20
  dBm** ([HopeRF datasheet v2.0](https://cdn.sparkfun.com/assets/a/9/6/1/0/RFM95W-V2.0.pdf)).
  Breakout ~$20 ([Adafruit](https://www.adafruit.com/product/3072)); bare
  module ~$8–16 in small quantity.
- **SX1262** (the chip current Meshtastic hardware — Heltec V3, RAK, Wio —
  has standardized on): SPI, RX ~4.2 mA, sleep as low as **1.62 µA**
  ([Seeed Wio-SX1262 datasheet](https://files.seeedstudio.com/products/SenseCAP/Wio_SX1262/Wio-SX1262_Module_Datasheet.pdf)),
  up to +22 dBm TX. Similar $10–15 pricing.

**Has anyone put a Teensy on a Meshtastic mesh?** No confirmed case. A
[PJRC forum thread](https://forum.pjrc.com/index.php?threads/teensy-4-1-and-lora-rmf95w.73036/)
shows someone wiring an RFM95W to a Teensy 4.1, but with no evidence of a
finished, working node. The
[official Meshtastic hardware list](https://meshtastic.org/docs/hardware/devices/)
has zero Cortex-M7 entries — every real device runs ESP32, nRF52840, or
RP2040/2350. Porting the firmware to the Teensy would be a from-scratch
bring-up, not a config change.

**Is Meshtastic open enough to interoperate?** Structurally yes: firmware,
apps, and protobufs are GPL-3.0
([firmware/LICENSE](https://github.com/meshtastic/firmware/blob/master/LICENSE)),
and the mesh-flooding algorithm and packet format are documented
([mesh-algo docs](https://meshtastic.org/docs/overview/mesh-algo/)). A
from-scratch device that correctly implements the LoRa modem parameters,
framing, and (for private channels) AES-256/Curve25519 could genuinely
interoperate with real Meshtastic nodes in the field. But this means
committing to reimplementing an actively-changing GPL-3.0 stack on an
unsupported MCU family — a sustained engineering burden, not a weekend
project. The newer alternative, **MeshCore**, is MIT-licensed but
**explicitly not interoperable** with Meshtastic — different packet format,
different routing, no bridge
([comparison](https://hexaspot.com/blogs/news/meshtastic-vs-meshcore-explained-same-hardware-different-firmware)).
Building a private, non-interoperable mesh instead is far cheaper but
forfeits the entire point of joining a mesh.

### 2. The bandwidth numbers that decide everything

Meshtastic's usable application payload is **237 bytes** per packet (256
byte frame minus protobuf/header overhead), confirmed across
[Meshtastic Discourse](https://meshtastic.discourse.group/t/meshtastic-lora-packet-size/7953)
and the [official overview docs](https://meshtastic.org/docs/overview/).

Airtime (Semtech AN1200.13 time-on-air, cross-checked against Meshtastic's
own published reference of 354 ms for a 16-byte SF11/250 kHz packet — see
[Meshtastic's own blog post on switching presets](https://meshtastic.org/blog/why-your-mesh-should-switch-from-longfast/)):

| Preset | 20 B | 50 B | 237 B (max) |
|---|---|---|---|
| SF7/125 kHz (ShortFast) | 41 ms | 82 ms | 349 ms |
| SF9/250 kHz | 107 ms | 184 ms | 698 ms |
| **SF11/250 kHz (LongFast, default)** | 246 ms | 380 ms | **1267 ms** |
| SF12/125 kHz (LongSlow) | 400 ms | 658 ms | 2125 ms |

Airtime at small payload sizes is dominated by fixed preamble/header
overhead, not the bytes themselves — this matters directly for section 3.

**US regulation is not the EU regime.** LoRa in US 915 MHz ISM is certified
under FCC Part 15.247's Digital Transmission System (DTS) path: ≥500 kHz
occupied bandwidth, ≤1 W conducted, PSD ≤+8 dBm/3 kHz
([15.247 guide](https://www.sunfiretesting.com/LoRa-FCC-Certification-Guide/),
[47 CFR 15.247](https://www.ecfr.gov/current/title-47/chapter-I/subchapter-A/part-15/subpart-C/subject-group-ECFR2f2e5828339709e/section-15.247)).
The dwell-time/channel-occupancy cap applies only to frequency-hopping
systems, and no FCC duty-cycle cap applies to DTS devices — confirmed by
Meshtastic's own firmware, which runs the US region at **unrestricted duty
cycle** while EU_433/EU_868 enforce a firmware-level **10% hourly cap**
because ETSI EN 300 220 requires it
([Meshtastic radio config docs](https://meshtastic.org/docs/configuration/radio/lora/)).
The only hard, region-independent firmware throttle is that Meshtastic
stops sending ancillary telemetry once measured channel utilization exceeds
40% — a channel-health safeguard, not a per-node byte budget, and it
doesn't gate user text messages.

**Bytes per hour a US node can legally send:** back-to-back 237-byte
LongFast packets (1267 ms each) → ~2,841 packets/hour ≈ **~673 KB/hour**,
legally. Applying the community's borrowed EU-style 10%-duty courtesy norm
as a *self-imposed* polite ceiling instead: 360 s airtime/hour ÷ 1.267
s/packet ≈ 284 packets/hour ≈ **~67 KB/hour**. Either way, this is a
generous budget relative to short text messages — the constraint that
matters is airtime-per-message latency and shared-channel courtesy, not a
legal cap.

### 3. The compression angle — the arithmetic

DeepMind's "Language Modeling Is Compression" ([arXiv 2309.10668](https://arxiv.org/abs/2309.10668),
ICLR 2024) gives raw enwik9 ratios by model size:

| Model | enwik9 ratio |
|---|---|
| Transformer 200K | 30.9% |
| Transformer 800K | 21.7% |
| Transformer 3.2M | 17.0% |
| Chinchilla 1B | 11.3% |
| Chinchilla 70B | 8.3% |
| gzip | 32.3% |
| LZMA2 | 23.0% |

Log-interpolating to ~30M params suggests ~15% — beating LZMA2 **on
paper**. Two things break this for the actual use case: (1) the paper notes
compression rate degrades sharply on short sequences, and a 100–300 byte
message gets almost none of the in-context adaptation benefit that produces
these headline numbers; (2) a direct 2026 empirical test contradicts the
extrapolation outright. [Haddadi & Teahan, EACL 2026](https://aclanthology.org/2026.eacl-srw.16.pdf)
ran LLaMA-3.2-**1B** (20–35× larger than this project's model) through a
FineZip-style arithmetic-coding pipeline on enwik8:

| Method | Ratio | Time (100 MB) |
|---|---|---|
| Tawa-td (trained PPM) | **0.2151 (best)** | minutes |
| lzma | 0.2478 | minutes |
| bz2 | 0.2896 | minutes |
| **LLM (LLaMA-3.2-1B)** | **0.2799** | **~12 hours** |
| zlib | 0.3649 | minutes |

A 1B-parameter model loses to plain lzma and is badly beaten by classical
PPM, at ~150× the compute time. A 30–57M model has no credible path to
beating lzma, let alone a near-zero-cost baseline.

**Compute cost on this hardware.** Arithmetic coding needs a full forward
pass per token on both encode and decode (same cost as generation, no
batched prefill). At the measured 15M-model rate (335.6 ms/token) and a
~30M model at roughly half that rate (671 ms/token, consistent with
memory-bandwidth-bound scaling), a ~150-byte message (~38 tokens at ~4
bytes/token) costs:

- 15M model: ~12.75 s encode + ~12.75 s decode ≈ **~25.5 s round trip**
- 30M model: ~25.5 s encode + ~25.5 s decode ≈ **~51 s round trip**

**Airtime actually at stake:** a raw 150-byte message at LongFast ≈ **~850
ms**. Even compressing to zero bytes saves at most 0.85 s.

**Verdict: net loss, decisively.** 25–51 s of added compute latency to save
≤0.85 s of airtime is a 30–100× mismatch in the wrong direction — corroborated
at datacenter-GPU scale by [Kipf et al., CIDR 2026](https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf),
who find LLM-based compression needs 10–120 years to amortize its compute
cost even on an L40S/A6000-class GPU, because inference latency dominates
even after batching and KV-cache optimization. The honest classical
baseline for 100–200 byte strings — **SMAZ**, a 253-entry static dictionary
compressor purpose-built for short ASCII text, reaching ~40–60% of original
size in microseconds with zero PSRAM traffic
([antirez/smaz](https://github.com/antirez/smaz)) — lands in the same
ratio ballpark as a tiny LM would plausibly achieve, for compute cost that
rounds to zero.

### 4. Other mesh-flavored uses

Judged against "does it reduce bytes-on-air":

- **Summarizing mesh traffic/logs for local display** — doesn't touch
  airtime (nothing new is transmitted); a legitimate offline-UX feature if
  25–50 s of local, non-blocking latency for an occasional "catch me up"
  query is acceptable, which it plausibly is on an otherwise-idle
  battery-powered node.
- **Structured messages from terse keywords, to save airtime** — a fixed
  set of message templates + a few slot bytes achieves the identical
  airtime reduction with zero inference cost. The LLM version is strictly
  worse here.
- **NL front end to node commands/configuration** — doesn't touch airtime;
  Meshtastic-class hardware is normally configured via a companion phone
  app over BLE, which already solves this more cheaply.
- **Store-and-forward digesting** — same category as the first bullet: a
  local convenience, not an airtime saver.

None of these clearly reduce bytes-on-air except the templated-message
idea, which needs no LLM. The survivors are justified purely as offline
human-facing convenience, weighed against tens of seconds of added latency
and non-trivial power draw on a battery-powered mesh node.

### 5. Prior art

**LLM + Meshtastic bridges exist, but none run inference co-resident with
the radio.** [mesh-ai](https://github.com/mr-tbot/mesh-ai),
[mesh-api](https://github.com/mr-tbot/mesh-api),
[radio-llm](https://github.com/pham-tuan-binh/radio-llm) (discussed on
[Hacker News](https://news.ycombinator.com/item?id=42429968)),
[llm-meshtastic-bridge](https://github.com/fiquett/llm-meshtastic-bridge),
and [MeshClaw](https://github.com/Seeed-Solution/MeshClaw) all run the LLM
on a phone/PC/SBC via Ollama/LM Studio/a cloud API, with the Meshtastic
node used purely as dumb LoRa transport. **No project puts an LLM on the
same constrained MCU that drives the radio.**

**LLM-based compression on embedded/MCU hardware: none found.** LLMZip
([arXiv 2306.04050](https://arxiv.org/abs/2306.04050), 9.5 days for 10 MB
on Llama-7B), FineZip ([arXiv 2409.17141](https://arxiv.org/abs/2409.17141),
needs an A6000 and 4 hours for 100 MB even after a 54× speedup), and the
CIDR/EACL papers above all run on GPUs or full desktop/server CPUs — strong
converging evidence nobody has found this practical even on hardware
100–1000× more capable than a Teensy 4.1.

### Synthesis — Candidate A

**Strongest honest version:** Drop compression as the headline feature.
Build a real (or at least protocol-compatible) LoRa node on an SX1262
module (~$10–15, 1.62 µA sleep), and use the on-board LLM purely as an
offline, local, human-facing convenience — summarizing accumulated mesh
traffic or answering questions about it when no phone/internet is present
(disaster response, off-grid base camp). This respects the airtime
economics because it never puts the LLM on the airtime-critical path.

**Killer objection:** the Teensy's scarce resource (tens of seconds of
PSRAM-bound compute per LM call) and LoRa's scarce resource (milliseconds
of airtime) point in opposite directions. Any version of "LLM helps the
mesh" either tries to save airtime — where it loses by 30–100× to
near-zero-cost classical compression — or doesn't touch airtime at all, in
which case it's an offline chatbot bolted onto a radio for no functional
reason connected to the radio's purpose.

**Compression verdict:** net loss, and not close. 25–51 s round-trip
compute versus ≤0.85 s of airtime saved, a mismatch corroborated at
datacenter-GPU scale, plus no credible evidence a ~30M model beats even
lzma on English text, let alone a free dictionary compressor.

---

## Candidate B — offline navigation / mapping aid

### 1. Data volumes

- **OSM planet** (compressed PBF, 2026-07-31 snapshot): **88 GB**
  ([planet.openstreetmap.org/pbf](https://planet.openstreetmap.org/pbf/)).
  Uncompressed: **~2.25 TB**
  ([OSM wiki](https://wiki.openstreetmap.org/wiki/Planet.osm)).
- **North America extract:** 17.9 GB PBF; **United States:** 11.2 GB PBF
  ([Geofabrik](https://download.geofabrik.de/north-america.html)).
- **A mid-size state (Colorado):** 359 MB PBF, 672–688 MB as
  shapefile/GeoPackage ([Geofabrik](https://download.geofabrik.de/north-america/us/colorado.html)).
- **A city (Denver, via BBBike):** 59 MB full PBF; **6.6 MB** for a
  Garmin-style routing-only ("Ontrail") export already filtered to the
  road/trail network relevant for navigation
  ([BBBike](https://download.bbbike.org/osm/bbbike/Denver/)).
- **Filtered POI/water/trail/road-only dataset:** no single authoritative
  number exists, but bounding from real analogs — a whole-country routable
  Garmin map (roads + POIs + routing graph, no building polygons) runs
  ~3 GB total, implying ~60 MB/state for a routing-focused cut
  ([forum thread](https://community.openstreetmap.org/t/routable-osm-map-of-continental-usa-for-garmin/56741)).
  A single national forest (roughly 1–3% of a state's area) lands in the
  **low single-digit MB** range. **A usable, aggressively filtered
  single-state dataset is credibly 20–80 MB; a single forest/region is
  1–10 MB.**

### 2. Confirming the "front end, not fact store" framing, with numbers

Recent memorization-capacity research
([arXiv 2505.24832, "How much do language models memorize?"](https://arxiv.org/abs/2505.24832),
tested 500K–1.5B params, squarely covering this project's range) puts pure
fact-memorization capacity at **~3.6 bits/parameter** when generalization is
deliberately excluded; related work (Allen-Zhu & Li) suggests closer to
~2 bits/parameter in practice. Applied to this project's 30M-param budget:

- 30M params × 3.6 bits = **13.5 MB absolute ceiling**, if the entire model
  were dedicated to rote memorization with zero capacity for language.
- At 2 bits/param: **7.5 MB**.
- A real model must spend most of that capacity on grammar and language
  competence, so the practically available fact-storage budget is well
  under even this ceiling — plausibly under 1–2 MB.

Compare to section 1: a single national forest's filtered data (1–10 MB)
already approaches or exceeds the model's entire theoretical memorization
ceiling, before subtracting anything for the capacity needed just to speak
English. One state (20–80 MB filtered, 359 MB raw) exceeds it by 2–45×. The
full US (11.2 GB) or planet (88 GB) exceed it by three to four orders of
magnitude. **The owner's own skepticism is correct and the numbers make it
not close: a 30M model cannot be a map database at any useful precision.**
It must be a front end over real data on the SD card.

### 3. Spatial indexing in ~100 KB of RAM

- **Tiled/gridded index** — the most practical option for this hardware.
  Fixed-size lat/lon cells, records sorted by cell at build time, a small
  in-RAM offset table (cell ID → SD byte offset + count). Query = compute
  covering cell(s), seek, bounded linear scan. RAM cost is a few KB to tens
  of KB depending on grid resolution; latency is dominated by predictable
  SD seeks. This is the standard approach in resource-constrained embedded
  GIS.
- **Geohash** — real embeddable C99 exists:
  [skeeto/geohash](https://github.com/skeeto/geohash) (no libc dependency,
  built for embedded use) and
  [yinqiwen/geohash-int](https://github.com/yinqiwen/geohash-int). Same
  practical profile as the grid approach, directly composable with it.
- **R-tree on flash/SD** — a real, studied problem: naive R-trees perform
  badly on flash because node updates cause write amplification
  ([Wu, Chang & Kuo](https://dl.acm.org/doi/10.1145/956676.956679)). Since
  this device's map data is **read-only at runtime** (built once, flashed
  to SD, never mutated in the field), the write-amplification concern that
  motivates that literature mostly doesn't apply — a static, offline-built
  R-tree or simple sorted grid read only from SD is a much easier target.
- **SQLite + R-Tree module** — SQLite itself genuinely runs on ESP32-class
  MCUs with SD-backed storage today (~500 KB RAM, 10M-row indexed lookups
  in ~700 ms,
  [siara-cc/esp32_arduino_sqlite3_lib](https://github.com/siara-cc/esp32_arduino_sqlite3_lib)),
  but that RAM figure is 5× this project's ~100 KB budget, and no
  confirmed report of the R-Tree module specifically running on an MCU
  exists. Plausible, unconfirmed.
- **H3, k-d tree** — real, portable C implementations exist
  ([uber/h3](https://github.com/uber/h3),
  [gishi523/kd-tree](https://github.com/gishi523/kd-tree)) but no
  MCU-specific footprint numbers were found for either.

**Verdict:** grid/geohash-bucket indexing is the only option with directly
confirmed embedded-C precedent at the right RAM budget; R-tree and
SQLite-R-Tree are architecturally plausible given read-only data but
unconfirmed at 100 KB and would need to be built and measured.

### 4. Does the NL layer earn its keep? (killer objection)

This device's own measured numbers make the objection quantitative, not
just a UX hunch: 309 ms time-to-first-token, 2.98 tok/s decode, **and no
batched prefill — every prompt token costs a full ~330 ms forward pass.**
For a realistic compound query ("find a campsite near water but not near a
road," ~12–18 tokens):

- Prompt ingestion at ~330 ms/token, unbatched: **~4–6 s** before any output
  exists.
- Generating a structured intent+slot output (~15–25 tokens) at 2.98 tok/s:
  **~5–8 s** more.
- **Total: ~10–13 s from typed query to structured output**, before the
  spatial-index lookup itself even runs.

A keyword/fuzzy-match system against the identical SD-resident index
(Levenshtein-over-trie implementations are memory-efficient and embeddable
in C — e.g.
[stevehanov.ca fuzzy-trie writeup](https://stevehanov.ca/blog/?id=114))
runs in **well under a second**. That is a 10–20× latency penalty on every
single query.

**When does NL genuinely help, independent of latency?**
- *Hands-busy/gloved-hands voice input* — the most common justification for
  NL interfaces in field gear — is **unavailable on this device**, which
  has no microphone and only USB-serial/keyboard-style text input. Adding
  voice would require a mic, an audio front end, and likely a much larger
  model for speech-to-text — a separate project, not a rounding error.
- *Compound/negated queries* ("near water but not near a road") — the
  honest strong case: a flat menu cannot express a three-clause
  conjunctive/negated spatial constraint without either a deep menu tree or
  collapsing to a cruder single-facet search. Real, but it pays the 10–13 s
  tax every time.
- *Users who don't know the taxonomy* — a well-designed fuzzy/autocomplete
  keyword search covers most of this without a generative model.

HCI literature is mixed, not a clean win for NL: a direct chatbot-vs-menu
study found chatbots rated **lower** on perceived user autonomy/control
([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0747563221004167));
for users who already know the query structure, NL-to-query interfaces
showed no efficiency advantage
([arXiv 2511.14718](https://arxiv.org/pdf/2511.14718)); a 1982 SIGCHI paper
on menu-constrained NLU essentially made this same tradeoff decades ago
([ACM](https://dl.acm.org/doi/10.1145/800045.801601)).

**Honest, non-hedging verdict:** On this specific hardware, the NL layer
does **not** clear the bar for the majority of queries (simple single-facet
lookups like "water near me," which keyword search handles just as well,
near-instantly). It only barely earns its keep for the minority of
genuinely compound/negated queries, and even there the 10+ second tax is
paid every time, not once. A menu/fuzzy-search system with a handful of
canned compound-filter presets would likely capture most of the real value
at a fraction of the latency and engineering risk.

### 5. The GPS question

Not required to prove the concept (manual lat/lon or place-name entry works
over the existing USB-serial channel), but required for genuine field
usefulness — "find water within 2 miles" is meaningless without a position,
and hand-typing coordinates every time defeats the purpose.

| Module | Price | Current draw | Notes |
|---|---|---|---|
| u-blox NEO-6M (GY-NEO6MV2) | $4.49–$15.99 | 45 mA tracking, 11 mA power-save | GPS only, cheapest ([u-blox datasheet](https://content.u-blox.com/sites/default/files/products/documents/NEO-6_DataSheet_(GPS.G6-HW-09005).pdf)) |
| Adafruit Ultimate GPS (MTK3339) | ~$25–40 | 20 mA | PPS output, common Teensy pairing ([Adafruit](https://www.adafruit.com/product/746)) |
| u-blox NEO-M9N (SparkFun Qwiic) | ~$55–77 | ~31 mA | 4-constellation GNSS, better fix time ([SparkFun](https://www.sparkfun.com/sparkfun-gps-breakout-neo-m9n-chip-antenna-qwiic.html)) |

An always-on GPS at 20–45 mA is not negligible on a battery device — for
scale, a Garmin inReach Mini 2 (a much heavier GPS+satellite power budget)
gets 14 days at default tracking, 30 in power-save, from its internal
battery ([review](https://gearjunkie.com/technology/gadgets/garmin-inreach-mini-2-review)).
Given this device's compute is already idle 92% of every token (a
low-power workload by construction), a continuously-tracking GPS module
could easily become the dominant power draw rather than a rounding error.
Firmware complexity is modest — NMEA-0183 parsing over UART is a few
hundred lines of well-trodden C — but the ~24 s cold-start means the device
is useless-for-position for up to 24 s after power-on, which matters for a
grab-and-go tool.

### 6. Prior art

**Commercial low-end devices already validate the SD-resident-OSM
approach:** the Garmin eTrex 10 has only 8–10 MB onboard storage, yet
OSM-derived topo maps with contours fit in 3.5–5 MB for a large region when
substituted for stock Garmin cartography
([writeup](http://captainbodgit.blogspot.com/2016/12/garmin-etrex-10-adding-topo-maps.html)) —
directly confirming that a filtered-OSM-on-microSD approach is not a novel
risk, just an established trick at commercial GPS-unit scale.

**LLM + maps at NL-front-end scale already exists — at server scale, not
edge:**
- **SPOT/TRIDENT** (demo at findthatspot.io,
  [community thread](https://community.openstreetmap.org/t/new-natural-language-search-interface-for-osm-public-demo-open-source/134816)):
  NL → Overpass QL over live OSM, fine-tuned **Mistral 24B** on a 24 GB
  GPU server — 800× this project's model size.
- **OverpassNL** ([TACL 2024](https://aclanthology.org/2024.tacl-1.31/),
  [code](https://github.com/raphael-sch/OverpassNL)): fine-tuned
  ByT5-base against a 306 GB OSM database replica, 8,352 NL↔query pairs
  plus 6,000 synthesized augmentations. Research-computing scale.
- **Hobbyist offline-map-on-MCU projects are real and numerous, and none
  add an LLM:** IceNav, PocketNav 32, Backcountry Beacon
  ([hackaday.io](https://hackaday.io/project/197411-icenav),
  [hackaday.io](https://hackaday.io/project/199321-pocketnav-32),
  [hackaday.io](https://hackaday.io/project/197362-backcountry-beacon)) —
  all ESP32 + GPS + SD-card OSM vector maps, all menu/button-driven.

**What's genuinely novel:** an NL front end small enough (~30M params) to
run fully offline on the *same* microcontroller as the map data, with zero
cloud/GPU dependency anywhere in the loop. No prior art found for that
specific combination — the gap is real, but section 4 shows novelty and
usefulness are separate questions here.

### 7. Training data for the fine-tune

Reference points for intent+slot semantic parsing at small scale:
- **SNIPS:** 15,884 utterances (13,084 train), 7 intents, 72 slots, >99%
  intent accuracy achievable with models far smaller than 30M params
  ([dataset](https://github.com/BrownFortress/IntentSlotDatasets)).
- **ATIS:** ~4,978 train / 893 test, 18–21 intents, 120–127 slots — the
  classic small-model NLU benchmark
  ([survey](https://arxiv.org/pdf/2101.08091)).
- Both suggest **5,000–16,000 examples** is sufficient for >95% accuracy on
  this class of task, historically with sub-30M-param models.
- **Counter-evidence, not to be ignored:** a 2024 study fine-tuning
  T5-Small (60.5M params — larger than this project's model) on
  structured-query generation with only 53–105 examples found it generated
  **not a single correct query**
  ([arXiv 2405.17076](https://arxiv.org/html/2405.17076v1)). The failure
  mode there was open-vocabulary entity linking against an arbitrary
  knowledge graph — structurally harder than a closed intent/slot schema —
  but it's a real warning that task difficulty, not just example count,
  determines convergence.

**Concrete plan:** generate templated (intent × slot-value) query/output
pairs, then use the owner's $10K AWS/Azure credits to call a large hosted
model for paraphrase/naturalization — the same methodology OverpassNL used
at larger scope. Target 5,000–15,000 examples. Cost: low thousands of
dollars for the paraphrase-generation API calls, a small fraction of
available credit. Fine-tuning a 30M model on that scale is trivial compute
— hours on a single Apple Silicon Mac or homelab GPU, not the bottleneck.
**This is feasible and well-matched to the owner's available compute; the
bottleneck is schema design and template coverage, not budget or GPU time.**

### Synthesis — Candidate B

**Strongest honest version:** a ~30M-param model fine-tuned on 5,000–15,000
synthetic (NL query → intent+slot JSON) pairs, running entirely on-device,
translating a small closed set of navigation intents (find POI by type,
get distance/direction, describe a route segment) into a structured query
against a grid/geohash-bucketed index of a pre-filtered, single-region OSM
extract (one state or forest, tens of MB) on the SD card. It never claims
to know a geographic fact itself — every answer is a live lookup — and its
only job is parsing phrasing a flat menu can't express cleanly.

**Killer objection:** this device's own measured numbers (no batched
prefill, ~330 ms/prompt-token, 2.98 tok/s decode) mean a single NL query
costs an estimated 10–13 s end to end, against well under a second for
fuzzy-keyword/menu search on the identical index. The majority of realistic
queries are simple single-facet lookups that gain nothing from NL and pay
the full latency tax. The device's marquee justification for NL in field
gear — voice input for gloved/hands-busy use — is structurally unavailable
(no microphone).

**Verdict:** the NL front end does not clearly earn its keep on this
hardware today. A keyword/fuzzy-search-plus-menu system over the same
spatial index serves the large majority of queries as well or better, at a
fraction of the latency and none of the training risk. The exception —
genuinely compound/negated queries — is real but narrow, and its value has
to be weighed against a 10+ second cost paid on every query, not just the
compound ones, unless the two paths are offered side by side.

---

## Scores and one-line verdicts

| | Technical feasibility | Story quality | Novelty |
|---|---|---|---|
| **A — LoRa mesh companion** | 5/10 — SPI hardware is trivial; a private mesh + local-LLM-convenience build is straightforward, but genuine Meshtastic protocol interop is a sustained GPL-3.0 port to an unsupported MCU family | 6/10 — "LLM on a mesh radio" pitches well, but the headline mechanism (compression) is decisively debunked by the numbers | 7/10 — no prior art found for an LLM co-resident with the radio MCU itself |
| **B — offline navigation aid** | 7/10 — every component (grid/geohash index, SD-resident filtered OSM, GPS module, synthetic fine-tune data) is well-understood and buildable with the owner's compute | 4/10 — appealing narrative, but the device's own measured latency numbers undercut the core UX claim | 7/10 — no prior art for an NL-to-map front end this small running fully offline on the same MCU as the data |

**Candidate A: pursue-in-modified-form.** Drop LM-based compression
entirely — it loses to sending raw bytes by 30–100× on latency and likely
loses on ratio too. Build the LoRa radio for its own sake (a private or,
if the GPL-3.0 stack is actually ported, interoperable mesh) and keep the
LLM off the airtime-critical path, using it only as an offline local
convenience (traffic digest, log summarization) that never has to justify
itself against airtime economics.

**Candidate B: pursue-in-modified-form.** Keep the "front end, not fact
store" architecture — the numbers prove it's the only viable one — but
ship it with a keyword/fuzzy-search-plus-menu interface as the primary
path, and gate the NL layer to an explicit opt-in mode for compound/negated
queries only, since it costs 10–13 s per query on this hardware and most
queries don't need it. Add GPS as a separate, explicitly-budgeted power
line item, not an assumed freebie.
