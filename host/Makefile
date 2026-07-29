# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
#
# Host build: the same core the Teensy runs, compiled three ways.
#
#   make test       native x86-64, generic C99 kernels
#   make test-arm   armv7-a + DSP under qemu-arm — the SMLAD kernels actually
#                   execute here, so the Cortex-M7 path is verified without
#                   hardware
#   make size-m7    bare-metal Cortex-M7 build, reports .text/.data footprint
#   make run        interactive generation against a model file

CORE_DIR  := ../core
CORE_SRC  := $(wildcard $(CORE_DIR)/src/*.c)
INC       := -I$(CORE_DIR)/include -I$(CORE_DIR)/src -Itests -Ibuild
WARN      := -std=c99 -Wall -Wextra -Werror -Wno-unused-parameter
CFLAGS    ?= -O2 -g $(WARN) $(INC)
LDLIBS    := -lm

PY        := PYTHONPATH=../tools python3
BUILD     := build

.PHONY: all test test-arm run fixtures golden size-m7 clean

all: $(BUILD)/tq_run $(BUILD)/test_all

# ------------------------------------------------------------------ fixtures

fixtures: $(BUILD)/test_q4.etq $(BUILD)/test_q8.etq

$(BUILD)/test_q4.etq: ../tools/etq/make_test_model.py ../tools/etq/format.py \
                      ../tools/etq/quantize.py ../tools/etq/convert.py | $(BUILD)
	$(PY) -m etq.make_test_model $@

$(BUILD)/test_q8.etq: ../tools/etq/make_test_model.py ../tools/etq/format.py \
                      ../tools/etq/quantize.py ../tools/etq/convert.py | $(BUILD)
	$(PY) -m etq.make_test_model --q8 $@

# golden.h is committed; this target regenerates it after a numerics change.
golden: $(BUILD)/test_q4.etq $(BUILD)/test_q8.etq
	$(PY) ../tools/gen_golden.py --model $(BUILD)/test_q4.etq \
		--q8-model $(BUILD)/test_q8.etq --out tests/golden.h

# --------------------------------------------------------------- native build

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/tq_run: tq_run.c $(CORE_SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_all: tests/test_all.c tests/golden.h $(CORE_SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/test_all.c $(CORE_SRC) $(LDLIBS)

test: $(BUILD)/test_all fixtures
	./$(BUILD)/test_all $(BUILD)/test_q4.etq

run: $(BUILD)/tq_run fixtures
	./$(BUILD)/tq_run $(BUILD)/test_q4.etq -n 32 -t 0.8

# ------------------------------------------------------- ARM DSP under QEMU
#
# armv7-a defines __ARM_FEATURE_DSP exactly like the Cortex-M7 does, so the
# SSUB8/SXTB16/SMLAD kernels compile and run for real. Static linking keeps
# qemu-arm from needing an ARM sysroot at runtime.

ARM_CC    ?= arm-linux-gnueabihf-gcc
QEMU      ?= qemu-arm-static
ARM_FLAGS := -march=armv7-a -mfpu=vfpv3-d16 -mfloat-abi=hard -static

$(BUILD)/test_all.arm: tests/test_all.c tests/golden.h $(CORE_SRC) | $(BUILD)
	$(ARM_CC) $(ARM_FLAGS) -O2 $(WARN) $(INC) -o $@ tests/test_all.c \
		$(CORE_SRC) $(LDLIBS)

test-arm: $(BUILD)/test_all.arm fixtures
	@echo "-- verifying the DSP kernels are really in the binary"
	@arm-linux-gnueabihf-objdump -d $(BUILD)/test_all.arm | \
		grep -cE '\bsmlad\b' | xargs -I{} sh -c \
		'test {} -gt 0 || { echo "no smlad instructions emitted!"; exit 1; }; \
		 echo "   smlad instructions: {}"'
	$(QEMU) ./$(BUILD)/test_all.arm $(BUILD)/test_q4.etq

# ---------------------------------------------------------- Cortex-M7 sizing

M7_CC    ?= arm-none-eabi-gcc
M7_FLAGS := -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard \
            -Os -ffunction-sections -fdata-sections

size-m7: | $(BUILD)
	@for f in $(CORE_SRC); do \
		$(M7_CC) $(M7_FLAGS) -std=c99 -Wall -Wextra -Werror $(INC) \
			-c $$f -o $(BUILD)/$$(basename $$f .c).m7.o || exit 1; \
	done
	@echo "-- Cortex-M7 object sizes (-Os)"
	@arm-none-eabi-size -t $(BUILD)/*.m7.o
	@echo "-- confirming the DSP path was selected"
	@arm-none-eabi-objdump -d $(BUILD)/tq_kernels.m7.o | \
		grep -cE '\bsmlad\b' | xargs -I{} echo "   smlad instructions: {}"

clean:
	rm -rf $(BUILD)
