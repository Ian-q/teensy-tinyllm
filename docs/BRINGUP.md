# Bring-up

From a soldered board to generated text. Each step has a pass condition; do
not move on until it is met.

## 0. Prerequisites

- A Teensy 4.1 with PSRAM fitted ([SOLDERING.md](SOLDERING.md))
- A microSD card, FAT32 or exFAT
- `teensy_loader_cli` or the Teensy Loader GUI
- A Zephyr 4.0 workspace (below)

## 1. Set up the Zephyr workspace

This repo is a west manifest repo, so it pins the exact Zephyr revision CI
builds against.

```bash
python3 -m venv ~/.venv/zephyr
source ~/.venv/zephyr/bin/activate
pip install west

cd <parent-dir>
git clone https://github.com/Ian-q/teensy-tinyllm.git
west init -l teensy-tinyllm
west update
west zephyr-export
pip install -r zephyr/scripts/requirements-base.txt
west sdk install -t arm-zephyr-eabi
```

Pass condition: `west --version` runs and `zephyr/` exists beside the repo.

## 2. Run the host tests first

Before touching hardware, confirm the engine is sane on a machine where you
can debug it.

```bash
make -C host test
```

Pass condition: `554 checks, 0 failures`.

NOTE: This is not a formality. Every numerical bug this project can have shows
up here first, and finding one at this stage costs seconds instead of an
evening with a logic analyser.

## 3. Build the firmware

```bash
west build -b teensy41 -p always teensy-tinyllm/firmware/teensy41-tinyllm
```

Pass condition: `zephyr/zephyr.hex` is produced.

## 4. Flash

```bash
teensy_loader_cli --mcu=TEENSY41 -w -v build/zephyr/zephyr.hex
```

Press the Program button when prompted.

Pass condition: the loader reports a successful write.

## 5. Open the console

The Teensy enumerates as a USB CDC-ACM device once Zephyr boots — the same
physical port used for flashing, a different USB device.

```bash
screen /dev/cu.usbmodem* 115200      # macOS
tio /dev/ttyACM0                     # Linux
```

Pass condition: a `tinyllm>` prompt. Press Enter if the screen is blank.

## 6. Confirm the PSRAM

```
tinyllm> tinyllm psram
```

Pass condition: the reported capacity matches what you soldered.

WARNING: If this reports `no PSRAM detected`, stop. Nothing downstream will
work. Go to the diagnosis table in [SOLDERING.md](SOLDERING.md#diagnosing-what-you-get-instead).

## 7. Memtest the whole array

```
tinyllm> tinyllm psram test
```

This takes a minute or two for 32 MB and destroys the contents, which at this
point are nothing.

Pass condition: `PASS (0 bad words)`.

CAUTION: A partial pass is a fail. A single bad word means one data or address
line is marginal, and a model with one corrupted weight produces subtly wrong
output rather than an obvious error.

## 8. Find the fastest stable clock

```
tinyllm> tinyllm psram sweep
```

Pass condition: it settles at 105.6 MHz or above. Record the number.

Once you know it, set `CONFIG_TINYLLM_PSRAM_BOOT_CLOCK_INDEX` in
`firmware/teensy41-tinyllm/prj.conf` and rebuild, so every boot starts there.

## 9. Convert a model

On your workstation:

```bash
pip install numpy
# stories15M is the recommended first model: 8.5 MB at Q4, ~4 tok/s,
# and it produces coherent children's stories rather than gibberish.
python3 -m etq.convert llama2c stories15M.bin tokenizer.bin stories15M.etq --q4 -v
```

The checkpoints and `tokenizer.bin` come from the `llama2.c` project
(`karpathy/tinyllamas` on HuggingFace for the weights, `tokenizer.bin` from the
`llama2.c` repo).

Pass condition: `wrote stories15M.etq: 8.5 MB, 62 tensors`, and the `-v`
per-tensor relative error is a few percent.

## 10. Sanity-check the model on the host

```bash
./host/build/tq_run stories15M.etq -i "Once upon a time" -n 40
```

Pass condition: readable English. If the host produces garbage, the conversion
is wrong and no amount of hardware debugging will help.

## 11. Copy to the card and load

Copy `stories15M.etq` to the microSD root, insert it, then:

```
tinyllm> tinyllm load stories15M.etq
loading /SD:/stories15M.etq: 10% 20% ... 100% done  (8320 KB in 389 ms, 21388 KB/s)

dim 288  hidden 768  layers 6  heads 6/6  vocab 32000
quant q4_0  kernel cortex-m7-dsp  context 256  kv fp32
psram: weights 8320 KB + kv 3456 KB + scratch 256 KB
dtcm : activations 143 KB of 256 KB
```

Pass condition: `kernel cortex-m7-dsp`. If it says `generic-c99` the DSP
extension was not detected and you are leaving ~2× of compute headroom on the
table — harmless here, but it means the build is not what you think it is.

## 12. Generate

```
tinyllm> tinyllm gen Once upon a time
```

Pass condition: coherent text at roughly the rate
[PERFORMANCE.md](PERFORMANCE.md) predicts for your measured bandwidth, and the
reported `effective MB/s` close to what `psram bench` measured.

## 13. Optional — run a model larger than the board

```
tinyllm> tinyllm stream stories110M.etq -c 512
tinyllm> tinyllm gen -n 20 Once upon a time
```

62 MB of weights on a 32 MB board, streamed off the card at roughly 0.5 tok/s.
The tokenizer is unavailable in streaming mode (it needs random access), so
output is raw token ids.

## If something is wrong

| Symptom | Where to look |
|---|---|
| No `tinyllm>` prompt | USB CDC-ACM enumeration. Check `CONFIG_USB_DEVICE_STACK_NEXT=y`; the legacy stack does not work on the RT1062. |
| `no SD card` | Card seated, FAT32/exFAT, and `usdhc1` enabled in the overlay. |
| Loads but generates gibberish | Verify the same `.etq` on the host first (step 10). If the host is fine and the device is not, suspect PSRAM — re-run `psram test`. |
| Much slower than predicted | Compare `gen`'s effective MB/s against `psram bench`. A large gap means something in the loop is stalling; a matching low number means the clock sweep settled low. |
| BusFault on first access | `CONFIG_USERSPACE` must stay off — see [HARDWARE.md](HARDWARE.md#memory-map-as-the-firmware-uses-it). |
