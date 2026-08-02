# Sourcing 16 MB QSPI PSRAM for Teensy 4.1 (2x chips, 16 MB -> 32 MB upgrade)

Researched 2026-07-31. All prices/stock verified live via WebFetch/WebSearch on this date;
re-check before ordering since boutique-retailer stock counts are volatile (ProtoSupplies
shows single-digit stock and caps orders per customer).

## Recommendation

**Buy from SparkFun Electronics.** Part `IS66WVS16M8FBLL-104NLI`, $8.95 each, in stock,
US-based (Boulder, CO). Two chips = **$17.90** + shipping. Correct SOIC-8 (SOP-8) package,
QSPI/QPI interface, 128 Mbit (16 MB), dual-die-stack — matches the PJRC/ProtoSupplies
reference part exactly.
[Product page](https://www.sparkfun.com/16-mb-psram-ic-is66wvs16m8fbll.html)

## Options checked, ranked

| Source | Part | Price (2x) | Stock | Notes |
|---|---|---|---|---|
| **SparkFun** (recommended) | IS66WVS16M8FBLL-104NLI | $17.90 | In stock | US-based, SOIC-8, listing explicitly says "works with Teensy 4.1" |
| ProtoSupplies | IS66WVS16M8FBLL-104NLI | $19.90 | 7 in stock, capped at 4/customer | US-based; this is the shop that co-developed the 16 MB option with PJRC and the Teensy community; sells a Teensy41_psram_memtest tool alongside it; page explicitly documents the 104 MHz spec (tested to 120 MHz) and the Teensyduino 1.60+ requirement |
| PJRC store | — | — | Not stocked | pjrc.com/store/psram.html sells only the 8 MB chip (via SparkFun link) and now *links out to ProtoSupplies* for the 16 MB part rather than selling it directly. PJRC does not carry the 16 MB chip itself. |
| The Pi Hut (UK) | IS66WVS16M8FBLL-104NLI | £17.20 (~$22 USD) | In stock | Same exact part, but UK-based — customs/duty and longer transit for a US buyer. Use only as a backup if the two US sources above are both out of stock. |
| DigiKey | — | — | **Not carried.** Direct search for `IS66WVS16M8` returns zero results. | Do not confuse with `IS66WVH16M8*BLL` (parallel-interface PSRAM, 24-TFBGA package) which *does* show up in DigiKey/Mouser searches for similar-looking part numbers — wrong package (BGA, not SOIC-8) and wrong protocol (parallel, not QSPI). Not a substitute. |
| Mouser | — | — | **Not carried.** Only smaller-density ISSI QSPI PSRAM in the same family show up (e.g. `IS66WVS4M8BLL-104NLI`, 32 Mbit/4 MB, and `IS66WVS8M8FBLL` variants), not the 128 Mbit dual-die part. | A forum note from mid-2025 said this part "was not yet in production" per ISSI's own site — it appears ISSI is routing it through smaller specialty channels (SparkFun, ProtoSupplies, The Pi Hut) rather than the big line-card distributors as of this check. |
| Arrow / Newark | — | — | No listings found for this part number in either catalog search. | Same conclusion as DigiKey/Mouser. |
| Amazon | — | — | **No matching 16 MB listing exists at all.** The only Teensy-PSRAM result on Amazon ("Teensy PSRAM Chips (5 Pack)", ASIN B08CTMSXL5) is the **8 MB** part, not 16 MB. | Nothing to buy here for this task — but as a general caution, if a 16 MB-labeled listing does show up on Amazon later, treat small-IC listings there as higher counterfeit/relabeling risk than SparkFun/ProtoSupplies/PJRC-affiliated shops; verify seller is Amazon-fulfilled and check date codes before trusting it. |
| eBay / AliExpress | — | — | No listings found for the exact part number on either platform. | Not needed as a fallback — the two US boutique sources both have stock. If you do find a listing later, treat it with the standard counterfeit/relabeled-die caution for small ICs from marketplace sellers. |

## Alternative parts: is there a known-good substitute?

**Short answer: no.** ISSI's `IS66WVS16M8FBLL` is effectively sole-source for "16 MB, QSPI/QPI,
SOIC-8 208-mil" — nothing else on the market combines that density, interface, and package.

- **AP Memory APS12808L (128 Mbit)** — **not a substitute, reject it.** It's an OPI
  (Octal SPI) **DDR** "Xccela" PSRAM in a **miniBGA 24-ball package (6x8x1.2 mm)**, not
  SOIC-8. Two dealbreakers at once: wrong package (BGA can't be hand-soldered to the
  Teensy's SOIC-8 pads) and wrong protocol (octal DDR, not the quad-SPI/QPI that
  `psram_flexspi2.c` and the Teensy FlexSPI2 controller expect). AP Memory's *QSPI*
  (single-data-rate, quad-SPI) PSRAM line — the family `APS6404L` belongs to — tops out
  at 64 Mbit (8 MB) in USON-8/SOIC-8 packages. AP Memory has no 128 Mbit part in that
  footprint/protocol combination, so there's no AP Memory drop-in for the 16 MB slot.
- **Lyontek LY68L6400** — this is a real, community-vetted SOIC-8 QSPI/QPI PSRAM, and
  it's reportedly ID-compatible with the ESP-PSRAM64H (same read-ID response), so it's a
  fine substitute *for the existing 8 MB chip*. But it's only 64 Mbit (8 MB) — Lyontek does
  not appear to make a 128 Mbit (16 MB) SOIC-8 QSPI part. Not applicable here.
- No other vendor (Winbond, Adesto/Renesas, etc.) turned up a 128 Mbit QSPI SOIC-8
  208-mil part in this search.

### Driver implication

Because both candidate alternates are disqualified on package/protocol grounds, there is
no new JEDEC ID to add to `psram_flexspi2.c` for *this* upgrade — you'd be buying the same
ISSI part the driver was presumably already tested against (or its non-stacked sibling).

One caveat I could **not** fully verify: I was unable to pull ISSI's exact Read-ID / KGD
byte table for `IS66WVS16M8FBLL` from a live datasheet during this search (distributor
datasheet mirrors 403'd or didn't surface the table). It is the same `IS66WVS` family as
the 8 MB part already recognized as ISSI ID `0x5D9D` with density in bits 23:21, and being
a dual-die-stack of the same 64 Mbit die, it should report through the same manufacturer ID
with just a different density field — but confirm this against the actual datasheet (or
just plug one in and read back the ID) before assuming the existing decode logic needs zero
changes.

## Sources

- https://www.sparkfun.com/16-mb-psram-ic-is66wvs16m8fbll.html
- https://protosupplies.com/product/psram_16mb/
- https://protosupplies.com/16mb-psram-available/
- https://www.pjrc.com/store/psram.html
- https://thepihut.com/products/16mb-psram-chip-for-teensy-4-1
- https://www.amazon.com/Teensy-PSRAM-Chips-5-Pack/dp/B08CTMSXL5
- https://www.mouser.com/datasheet/2/1127/APM_PSRAM_OPI_Xccela_APS12808L_OBMx_v3_0a_PKG-1954889.pdf (AP Memory APS12808L datasheet — confirms OPI/DDR/miniBGA)
- https://www.lyontek.com.tw/pdf/ddr/LY68L6400-1.2.pdf (Lyontek LY68L6400 — confirms 64 Mbit/SOIC-8/QPI)
- DigiKey, Mouser, Arrow, Newark searches for `IS66WVS16M8` / `IS66WVS16M8FBLL` — no results
  (checked directly against each distributor's search)
