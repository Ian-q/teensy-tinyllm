# Hardware

## The two pads

The Teensy 4.1 has two unpopulated **SOIC-8 footprints on its underside**,
beneath the microSD socket, between the pin 31/32 and 33/34 rows. Both hang
off the RT1062's **FlexSPI2** controller, on separate chip selects:

| Footprint | Chip select | Pad | Intended part | AHB address |
|---|---|---|---|---|
| "RAM" | `FLEXSPI2_A_SS0_B` | `GPIO_EMC_24` | PSRAM | `0x70000000` |
| "Flash" | `FLEXSPI2_A_SS1_B` | `GPIO_EMC_22` | PSRAM **or** QSPI NOR | immediately after chip 1 |

The shared bus pins are `GPIO_EMC_25` (SCLK), `EMC_26..29` (DATA0–3) and
`EMC_23` (DQS).

The two chip selects are mapped back to back, so **two PSRAM parts appear as
one flat address range** starting at `0x70000000`. They do **not** double
bandwidth — one controller, one set of data lines, one clock.

## Three configurations

| | Config A — **MAXRAM** | Config B — XIP | Config C — minimal |
|---|---|---|---|
| CS0 | 16 MB PSRAM | 16 MB PSRAM | 8 MB PSRAM |
| CS1 | 16 MB PSRAM | 16 MB QSPI NOR | *(empty)* |
| Total addressable | **32 MB RAM** | 16 MB RAM + 16 MB flash | 8 MB RAM |
| Largest resident Q4 model | ~42M params | ~24M params | ~12M params |
| Model persists across power cycles | no (reload from SD, ~1 s) | yes (XIP) | no |
| Parts cost | ~$20 | ~$14 | ~$6 |

**Build Config A.** The premise of this project is "max out the board", and
32 MB of writable memory is strictly the most flexible thing you can put on
those two pads. The case for Config B is that the model lives in flash and
survives a power cycle, so there is no boot-time load — but the load is about
one second from SD, and giving up half your RAM to save one second is a bad
trade when RAM is the thing that caps model size. Config B also cannot hold
the KV cache and the weights in the same medium, which complicates the memory
plan for no benefit.

Config C is what you already have if you bought a Teensy 4.1 with the standard
PJRC 8 MB PSRAM add-on. It runs `stories15M` fine.

## Bill of materials — Config A

| Item | Part | Qty | Notes |
|---|---|---|---|
| MCU board | PJRC **Teensy 4.1** | 1 | i.MX RT1062, 600 MHz Cortex-M7, 1 MB on-chip RAM, 8 MB program flash |
| PSRAM | **ISSI IS66WVS16M8FBLL** (16 MB / 128 Mbit, SOIC-8 208-mil) | 2 | Two 8 MB dies per package. This is the part ProtoSupplies sells as "16MB PSRAM for Teensy 4.1". |
| microSD | any FAT32/exFAT card, ≥1 GB, class 10 | 1 | Holds the `.etq` models. Sequential read ~22 MB/s over 4-bit SDIO. |
| USB cable | micro-B, data-capable | 1 | Flashing and console are the same port. |

**8 MB alternative:** `APS6404L-3SQR` (AP Memory) or the equivalent Ipus / ESP
/ Lyontek part, which is what PJRC sells directly. Two of those give 16 MB
instead of 32 MB. The driver detects and reports either.

### Don't want to solder?

ProtoSupplies sells the **"Teensy 4.1 Fully Loaded"** with PSRAM already
installed, and sells the 16 MB parts individually. Buying a pre-populated
board is a completely legitimate way to skip the only genuinely difficult step
in this project.

## Clock and bandwidth

FlexSPI2 runs from `CCM_CBCMR` and the firmware can retune it at runtime.
The available steps, and what each implies:

| Index | Clock | Theoretical (4-bit SDR) | Realistic sustained read |
|---|---|---|---|
| 0 | 88.0 MHz | 44.0 MB/s | ~28 MB/s |
| 3 | 105.6 MHz | 52.8 MB/s | ~34 MB/s |
| 5 | 120.0 MHz | 60.0 MB/s | ~38 MB/s |
| 6 | 132.0 MHz | 66.0 MB/s | ~42 MB/s |
| 8 | 166.2 MHz | 83.1 MB/s | ~53 MB/s |

The ~65% efficiency against theoretical is command and turnaround overhead;
it is consistent with the ~28 MB/s figure the PJRC forum reports for a
`memcpy` from PSRAM at the 88 MHz default.

**88 MHz was the Teensyduino default until 1.60, which raised it to 105.6 MHz.**
Anything above that is per-board: it depends on the specific silicon and, more
than anything, on your solder joints. Hence `tinyllm psram sweep`, which
memtests at each step and settles on the fastest that passes. ProtoSupplies
reports the 16 MB parts running reliably at 120 MHz and the 8 MB parts as far
as 166 MHz — the 16 MB packages load the bus harder because there are two dies
behind one set of pins.

Token rate is directly proportional to this number. Going from the 88 MHz
default to a part that sweeps clean at 166 MHz is a **1.9× speedup for free**,
which is why the sweep exists rather than a hardcoded constant.

## Memory map as the firmware uses it

```
0x00000000  ITCM     512 KB   instructions (Zephyr text)
0x20000000  DTCM     512 KB   activations, logits, stacks   <- fast, single cycle
0x20200000  OCRAM    512 KB   Zephyr heap, DMA buffers
0x60000000  Flash      8 MB   program flash (FlexSPI1, onboard)
0x70000000  PSRAM     32 MB   model weights, KV cache, sampler scratch
            +0             model image           8-30 MB
            +model_bytes   KV cache               1-9 MB
            +...           sampler scratch         256 KB
```

`0x70000000` is reachable without any MPU configuration because Zephyr enables
`PRIVDEFENA` on ARMv7-M: privileged accesses that miss every MPU region fall
through to the architectural default map, where `0x60000000–0x9FFFFFFF` is
Normal, cacheable, write-back memory. That gives us both the caching we want
and legal unaligned access, which matters because Q4 blocks are 18 bytes and
therefore only 2-byte aligned half the time. **If `CONFIG_USERSPACE` is ever
enabled this stops being true** and an explicit region has to be declared.

## What is deliberately not used

- **SEMC / SDRAM.** The RT1062 has a 16-bit SDRAM controller that would beat
  QSPI on bandwidth, but PJRC does not route those pins on the Teensy 4.1.
  Not available at any price.
- **Overclocking the CPU.** The Teensy runs happily at 816 MHz and beyond, and
  it would buy essentially nothing: the core is already idle ~92% of the time
  waiting on PSRAM. Spend the thermal budget on the FlexSPI2 clock instead.
- **The second FlexSPI controller.** FlexSPI1 is committed to the onboard
  program flash. Splitting weights across both controllers to double bandwidth
  is theoretically interesting and practically means giving up the ability to
  boot.
