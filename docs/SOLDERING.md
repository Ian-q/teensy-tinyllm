# Soldering the PSRAM

Two SOIC-8 packages onto the underside of a Teensy 4.1. This is the only step
in the project that can destroy hardware, and it is genuinely the hard part.

> **You can skip this entirely.** ProtoSupplies sells the Teensy 4.1 "Fully
> Loaded" with PSRAM already installed. If you want the language model and not
> the soldering, buy that. Nothing downstream cares how the chips got there.

## Before you start

**Read the whole page first.** Steps 6 and 7 in particular change what you do
in step 4.

WARNING: The RT1062 and the microSD socket sit within a few millimetres of the
target pads. Hot air at 350 °C aimed carelessly will reflow the SD socket or
lift a passive. Shield everything you are not heating.

CAUTION: PSRAM packages are ESD sensitive. Wear a wrist strap, or at minimum
touch a grounded surface before handling the parts.

### Tools

- Fine-conical or small-chisel iron, temperature controlled, **330–350 °C**
- No-clean flux, in a pen or syringe — the single most important item here
- Thin solder, 0.5 mm or finer, leaded (Sn63/Pb37) if you have it; it flows at
  a lower temperature and is far more forgiving
- Solder wick, 1.5–2 mm
- Kapton tape
- 10× magnification or better
- Isopropyl alcohol and a stiff brush
- Multimeter with continuity beep
- Optional but excellent: hot-air station, low flow

## Procedure

### 1. Identify the footprints

Turn the board over. Between the pin 31/32 and 33/34 rows, under the microSD
socket, are two unpopulated SOIC-8 footprints. PJRC silkscreens them. One is
chip select 0 (the "RAM" position, pad `GPIO_EMC_24`); the other is chip
select 1 (the "Flash" position, `GPIO_EMC_22`).

NOTE: For a 32 MB build both footprints get a PSRAM chip. The chip-select
labels describe the intended part, not a restriction.

### 2. Mask the neighbours

Cover the microSD socket and any nearby components with Kapton tape.

### 3. Tin one corner pad

Apply flux to the footprint. Put a small amount of solder on **one** corner pad
only. Not all eight.

### 4. Place and tack the chip

Match the package's pin-1 dot to the pin-1 marker on the silkscreen.

WARNING: A chip soldered 180° out will not answer its JEDEC ID and, depending
on which pins land where, can short a supply rail. Confirm the orientation
under magnification before applying heat.

Hold the chip in place and reflow the single tinned corner pad. Check
alignment again — all eight leads centred on their pads — and correct now
while only one joint holds it.

### 5. Solder the remaining seven

Flux generously. Drag-solder or joint-by-joint, whichever you are better at.
Bridges are expected; wick them afterwards.

### 6. Wick and inspect

Remove every bridge with wick and fresh flux. Then inspect all eight joints at
10× or better. Look for a concave fillet on each lead. A lead that merely sits
on a shiny pad is not soldered.

### 7. Repeat for the second chip

### 8. Clean

Scrub with isopropyl alcohol and let it dry completely. Flux residue between
QSPI pins running at 132 MHz causes exactly the kind of intermittent failure
that is miserable to diagnose.

### 9. Continuity check, power off

Beep out every adjacent pin pair on each package. Any beep between neighbours
is a bridge — go back to step 6.

Then confirm each data line reaches the RT1062 and is not shorted to ground.

### 10. First power-on

Connect USB. Flash the firmware ([BRINGUP.md](BRINGUP.md)) and run:

```
tinyllm> tinyllm psram
```

Expected on a good Config A build:

```
32 MB across 2 chips @ 105.6 MHz
  CS0 (pad EMC_24): 16 MB, id 0x009d5d9d
  CS1 (pad EMC_22): 16 MB, id 0x009d5d9d
```

## Diagnosing what you get instead

| Symptom | Almost always means |
|---|---|
| `no PSRAM detected`, chip 1 id `0x00000000` | Open joint on `DATA0` (`EMC_26`) or the chip is unpowered. Reflow the whole package. |
| `no PSRAM detected`, id `0xffffffff` | Open on `SCLK` (`EMC_25`) or `CS` — the bus is floating high. |
| Only CS0 detected, CS1 expected | Open on `EMC_22` (the CS1 pad specifically). The shared bus is fine or chip 1 would have failed too. |
| ID reads but capacity is wrong | Wrong part fitted, or ID bits 23:21 are being corrupted — a marginal data line. Run `tinyllm psram test`. |
| Detects fine, `psram test` fails | Bridge or cold joint on a data line. Note *which* bits fail: a single stuck bit points straight at one `DATA` pin. |
| Passes at 88 MHz, fails above ~105 MHz | Joints are electrically connected but mechanically poor — excess resistance or stray capacitance. Reflow with fresh flux. This is the most common outcome of a rushed job. |
| Detects, then a BusFault on first access | Not a soldering problem. Check `CONFIG_USERSPACE=n` — see [HARDWARE.md](HARDWARE.md#memory-map-as-the-firmware-uses-it). |

A board that enumerates at 88 MHz but will not sweep past 105 MHz is still
fully usable — it just runs about 20% slower. Reflowing is worth one attempt,
not five.

## What "good" looks like

```
tinyllm> tinyllm psram sweep
 88.0 MHz: pass  read 28.4 MB/s
 99.0 MHz: pass  read 31.9 MB/s
102.9 MHz: pass  read 33.1 MB/s
105.6 MHz: pass  read 34.0 MB/s
110.8 MHz: pass  read 35.6 MB/s
120.0 MHz: pass  read 38.3 MB/s
132.0 MHz: pass  read 42.1 MB/s
144.0 MHz: FAIL
settled on 132.0 MHz
```

Once you know the number, set `CONFIG_TINYLLM_PSRAM_BOOT_CLOCK_INDEX` in
`prj.conf` so it applies from boot instead of needing a sweep every session.
