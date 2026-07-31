/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * Fault capture across a reboot.
 *
 * The only console on this board is USB CDC, and USB dies with the CPU: a
 * fault handler's output never reaches the host, so every crash looks like
 * a silent hang. Instead of printing, the handler stashes the register
 * state in a __noinit struct — SRAM survives a warm reset — and resets.
 * The next boot finds the magic and prints the crash that just happened.
 *
 * `tinyllm fault` re-prints the last capture; `tinyllm fault clear` forgets
 * it.
 */

#include <string.h>

#include <zephyr/arch/cpu.h>
#include <zephyr/fatal.h>
#include <zephyr/init.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>
#include <zephyr/sys/barrier.h>
#include <zephyr/sys/reboot.h>

LOG_MODULE_REGISTER(fault, CONFIG_TINYLLM_LOG_LEVEL);

#define FAULT_MAGIC 0x0FA17EDu

struct fault_record {
	uint32_t magic;
	uint32_t reason;
	uint32_t pc;
	uint32_t lr;
	uint32_t psr;
	uint32_t cfsr;    /* configurable fault status: which fault, and why */
	uint32_t hfsr;
	uint32_t bfar;    /* faulting data address, if BFARVALID            */
	uint32_t mmfar;
};

static __noinit struct fault_record last_fault;

void k_sys_fatal_error_handler(unsigned int reason, const struct arch_esf *esf)
{
	last_fault.magic  = FAULT_MAGIC;
	last_fault.reason = reason;
	last_fault.pc     = esf ? esf->basic.pc : 0u;
	last_fault.lr     = esf ? esf->basic.lr : 0u;
	last_fault.psr    = esf ? esf->basic.xpsr : 0u;
	last_fault.cfsr   = SCB->CFSR;
	last_fault.hfsr   = SCB->HFSR;
	last_fault.bfar   = SCB->BFAR;
	last_fault.mmfar  = SCB->MMFAR;

	/* Push the record out of the cache before the reset drops it. */
	barrier_dsync_fence_full();
	sys_reboot(SYS_REBOOT_WARM);
	CODE_UNREACHABLE;
}

static void fault_print(const struct shell *sh)
{
	uint32_t cfsr;

	if (last_fault.magic != FAULT_MAGIC) {
		shell_print(sh, "no fault recorded");
		return;
	}
	cfsr = last_fault.cfsr;
	shell_print(sh, "last fault: reason %u  pc 0x%08x  lr 0x%08x  psr 0x%08x",
		    last_fault.reason, last_fault.pc, last_fault.lr,
		    last_fault.psr);
	shell_print(sh, "  cfsr 0x%08x  hfsr 0x%08x  bfar 0x%08x  mmfar 0x%08x",
		    cfsr, last_fault.hfsr, last_fault.bfar, last_fault.mmfar);
	/* Decode the bits that actually distinguish our failure modes. */
	if (cfsr & (1u << 16)) {
		shell_print(sh, "  UNDEFINSTR — executed garbage");
	}
	if (cfsr & (1u << 24)) {
		shell_print(sh, "  UNALIGNED — unaligned access with STRICT_ALIGN");
	}
	if (cfsr & (1u << 25)) {
		shell_print(sh, "  DIVBYZERO");
	}
	if (cfsr & (1u << 8)) {
		shell_print(sh, "  IBUSERR — instruction fetch failed");
	}
	if (cfsr & (1u << 9)) {
		shell_print(sh, "  PRECISERR — bus error, address in BFAR");
	}
	if (cfsr & (1u << 10)) {
		shell_print(sh, "  IMPRECISERR — late bus error (buffered write)");
	}
	if (cfsr & (1u << 1)) {
		shell_print(sh, "  DACCVIOL — MPU denied a data access, addr in MMFAR");
	}
	if (cfsr & (1u << 0)) {
		shell_print(sh, "  IACCVIOL — MPU denied an instruction fetch");
	}
}

static int cmd_fault(const struct shell *sh, size_t argc, char **argv)
{
	if (argc > 1 && !strcmp(argv[1], "clear")) {
		last_fault.magic = 0u;
		shell_print(sh, "cleared");
		return 0;
	}
	fault_print(sh);
	return 0;
}

SHELL_CMD_REGISTER(fault, NULL, "show the fault that caused the last reboot",
		   cmd_fault);

static int fault_announce(void)
{
	if (last_fault.magic == FAULT_MAGIC) {
		LOG_ERR("rebooted by a fault: pc 0x%08x cfsr 0x%08x "
			"(run `fault` for detail)",
			last_fault.pc, last_fault.cfsr);
	}
	return 0;
}

SYS_INIT(fault_announce, APPLICATION, CONFIG_APPLICATION_INIT_PRIORITY);
