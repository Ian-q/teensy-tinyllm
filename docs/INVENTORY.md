# Workshop inventory

Hardware on hand for this project and its offshoots. Recorded here because a
chat log is a bad place to keep it, and because several of the ideas in the
issue tracker are gated on what is physically available rather than on any
software work.

Verified facts are marked as such. Anything unverified says so — this page
follows the same rule as [STATUS.md](STATUS.md): do not assert what has not
been checked.

## The board

| | |
|---|---|
| **Teensy 4.1** | i.MX RT1062, Cortex-M7 @ 600 MHz, 1 MB on-chip RAM, 8 MB flash |
| **PSRAM** | 2x APS6404L-3SQR hand-soldered to the underside pads. **16 MB, verified** — ids `0x32535d0d`, 31.4 MB/s sustained read at 105.6 MHz |
| **microSD** | FAT32, holds the `.etq` model files |
| **Console** | USB CDC-ACM, enumerates as `/dev/cu.usbmodem2101` |

Upgrade path to 32 MB is [#24](https://github.com/Ian-q/teensy-tinyllm/issues/24)
— 2x ISSI `IS66WVS16M8FBLL-104NLI` from SparkFun. See
[research/psram-sourcing.md](research/psram-sourcing.md) for prices, stock, and
the `IS66WVH`-vs-`IS66WVS` package trap that makes the wrong part unsolderable
to these pads.

## Radios

**2x Baofeng BF-F8HP-PRO**, 8 W tri-power VHF/UHF handhelds.

Relevant to [#29](https://github.com/Ian-q/teensy-tinyllm/issues/29) (Semaphore
over VHF/UHF). Two notes that shape any project using them:

- **Dual watch is not full duplex.** The UV-5R lineage time-slices one receiver
  between two VFOs and locks onto whichever gets traffic first. One receiver,
  one transmitter. Simultaneous transmit-here / receive-there needs two
  independent RF chains. *Unverified for the PRO specifically, which is newer —
  check whether the manual says "dual watch" or "true dual receive".* Having two
  radios is the practical answer regardless.
- **Transmitting needs a licence.** At least a Technician for 2 m / 70 cm.
  Digital modes are fine; see the Part 97.113(a)(4) discussion in #29 for why
  Semaphore's compression is a published code rather than prohibited
  obscuration, and why that depends on the repo staying public.

Interfacing is the standard soundcard-modem shape: Kenwood 2-pin, speaker out
to a Teensy ADC, PWM-and-filter back into the mic line, PTT via a transistor.

## Display

**Sony Watchman FD-20A**, manufactured June 1984. Flat B&W CRT — the design
with the electron gun mounted off-axis, firing across the faceplate rather than
from behind it. Serial 249594, FCC ID AK896AFD-20A.

**Verified from photographs and a power-on test:**

- Powers on, good cosmetic condition.
- **`VIDEO IN` on a 3.5 mm jack.** Composite goes straight in — no RF
  modulator, no opening the case, no exposure to the anode supply. This is what
  makes [#28](https://github.com/Ian-q/teensy-tinyllm/issues/28) tractable.
- **`V HOLD` and `CONTR` are external trim pots**, which matters more than it
  sounds: vertical hold adjustable from outside is the safety net for a
  hand-rolled NTSC generator with slightly off-spec field timing.
- `DC IN 6V` barrel, `EAR` jack, VHF/UHF band select, tuning and volume sliders.
- Receives nothing off-air. US analogue broadcast ended in 2009, which is
  exactly what makes it available for this.

**Unverified, check before wiring:**

- Whether the 3.5 mm jack is mono (tip = video) or TRS (tip = video, ring =
  audio). Meter it.
- **Barrel polarity on the 6 V input.** Vintage Sony portables were frequently
  centre-negative and getting it backwards could end the project.
- Condition of the electrolytics. 42 years is well past their design life, but
  recapping should follow a symptom rather than be done prophylactically.

## Development host

macOS on arm64. Zephyr workspace at `~/dev` with the SDK at
`~/zephyr-sdk-0.17.0`; `west` lives in `~/.venv/zephyr/bin`. Build locally — CI
is the push gate, not the compiler.

No ARM cross-compiler or `qemu-arm` natively: Homebrew's QEMU has no
`linux-user` targets on macOS, and Zephyr v4.0.0 ships no Cortex-M7 QEMU board.
`make -C host docker-test` runs the ARM suite in a container instead, which is
how `test-bitexact` gets exercised without pushing.
