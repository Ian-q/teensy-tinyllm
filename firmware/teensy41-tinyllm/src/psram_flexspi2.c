/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * FlexSPI2 PSRAM bring-up for the Teensy 4.1 bottom-side QSPI pads.
 *
 * Why this exists rather than Zephyr's memc_mcux_flexspi_aps6404l:
 *
 *   1. That driver does not build for teensy41 on Zephyr 4.0 — it assigns
 *      flexspi_device_config_t.addressShift, a field only the RW612 SoCs have
 *      (zephyr#83244, fixed upstream in #97380, after our pinned revision).
 *   2. It assumes ONE 8 MB APS6404L. This board can carry TWO parts, and the
 *      16 MB ISSI IS66WVS16M8 needs a different ID decode (two 8 MB dies
 *      behind one package, reported through JEDEC ID bits 23:21).
 *   3. We want to change the FlexSPI2 root clock at runtime and memtest at
 *      each step. Bandwidth is the entire performance story for LLM decode,
 *      and the stock driver fixes the clock at DT-configure time.
 *
 * Registers are declared here directly from the i.MX RT1062 reference manual
 * rather than pulled from the MCUX SDK headers, so this file has no dependency
 * beyond <stdint.h> and compiles identically regardless of which HAL revision
 * the Zephyr workspace happens to hold. Bring-up sequence, LUT contents and
 * the JEDEC ID decode follow PJRC's cores/teensy4/startup.c, which is the
 * reference implementation this hardware is known to work with.
 */

#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/barrier.h>

#include "psram_flexspi2.h"

LOG_MODULE_REGISTER(psram, CONFIG_TINYLLM_LOG_LEVEL);

/* ------------------------------------------------------------- registers */

#define REG32(a) (*(volatile uint32_t *)(uintptr_t)(a))

/* MIMXRT1062.h, pulled in through the Zephyr SoC headers, also defines the
 * three peripheral base addresses below — same values, different spelling,
 * which -Werror reports as a redefinition. Verify agreement, then take
 * ownership of the names: this file's register map is authoritative here
 * (decision D4 in docs/RESEARCH.md). */
#if defined(CCM_BASE) && (CCM_BASE != 0x400FC000u)
#error "SDK header disagrees with the PJRC-verified CCM base address"
#endif
#if defined(IOMUXC_BASE) && (IOMUXC_BASE != 0x401F8000u)
#error "SDK header disagrees with the PJRC-verified IOMUXC base address"
#endif
#if defined(FLEXSPI2_BASE) && (FLEXSPI2_BASE != 0x402A4000u)
#error "SDK header disagrees with the PJRC-verified FlexSPI2 base address"
#endif
#undef CCM_BASE
#undef IOMUXC_BASE
#undef FLEXSPI2_BASE

#define CCM_BASE      0x400FC000u
#define CCM_CBCMR     REG32(CCM_BASE + 0x018u)
#define CCM_CCGR7     REG32(CCM_BASE + 0x084u)

/* CCM analog (PLL/PFD) block. Our own names — the SDK header also covers
 * these registers. */
#define CCMA_BASE     0x400D8000u
#define CCMA_PFD_480  REG32(CCMA_BASE + 0x0F0u)
#define CCMA_PFD_528  REG32(CCMA_BASE + 0x100u)

#define IOMUXC_BASE   0x401F8000u
/* SW_MUX_CTL_PAD_GPIO_EMC_22 is at +0x06C; the eight pads we need are
 * consecutive 32-bit registers from there. Pad control mirrors them at
 * +0x25C. Input-select registers for FlexSPI2 sit in the second block. */
#define IOMUXC_MUX_EMC22  (IOMUXC_BASE + 0x06Cu)
#define IOMUXC_PAD_EMC22  (IOMUXC_BASE + 0x25Cu)
#define IOMUXC_SEL_BASE   (IOMUXC_BASE + 0x400u)
#define IOMUXC_SEL_DQS_FA   REG32(IOMUXC_SEL_BASE + 0x32Cu)
#define IOMUXC_SEL_IO_FA0   REG32(IOMUXC_SEL_BASE + 0x330u)
#define IOMUXC_SEL_IO_FA1   REG32(IOMUXC_SEL_BASE + 0x334u)
#define IOMUXC_SEL_IO_FA2   REG32(IOMUXC_SEL_BASE + 0x338u)
#define IOMUXC_SEL_IO_FA3   REG32(IOMUXC_SEL_BASE + 0x33Cu)
#define IOMUXC_SEL_SCK_FA   REG32(IOMUXC_SEL_BASE + 0x350u)

#define FLEXSPI2_BASE 0x402A4000u
#define FS2_MCR0        REG32(FLEXSPI2_BASE + 0x000u)
#define FS2_MCR1        REG32(FLEXSPI2_BASE + 0x004u)
#define FS2_MCR2        REG32(FLEXSPI2_BASE + 0x008u)
#define FS2_AHBCR       REG32(FLEXSPI2_BASE + 0x00Cu)
#define FS2_INTEN       REG32(FLEXSPI2_BASE + 0x010u)
#define FS2_INTR        REG32(FLEXSPI2_BASE + 0x014u)
#define FS2_LUTKEY      REG32(FLEXSPI2_BASE + 0x018u)
#define FS2_LUTCR       REG32(FLEXSPI2_BASE + 0x01Cu)
#define FS2_AHBRXBUFCR0(n) REG32(FLEXSPI2_BASE + 0x020u + 4u * (n))
#define FS2_FLSHA1CR0   REG32(FLEXSPI2_BASE + 0x060u)
#define FS2_FLSHA2CR0   REG32(FLEXSPI2_BASE + 0x064u)
#define FS2_FLSHA1CR1   REG32(FLEXSPI2_BASE + 0x070u)
#define FS2_FLSHA2CR1   REG32(FLEXSPI2_BASE + 0x074u)
#define FS2_FLSHA1CR2   REG32(FLEXSPI2_BASE + 0x080u)
#define FS2_FLSHA2CR2   REG32(FLEXSPI2_BASE + 0x084u)
#define FS2_IPCR0       REG32(FLEXSPI2_BASE + 0x0A0u)
#define FS2_IPCR1       REG32(FLEXSPI2_BASE + 0x0A4u)
#define FS2_IPCMD       REG32(FLEXSPI2_BASE + 0x0B0u)
#define FS2_IPRXFCR     REG32(FLEXSPI2_BASE + 0x0B8u)
#define FS2_IPTXFCR     REG32(FLEXSPI2_BASE + 0x0BCu)
#define FS2_RFDR0       REG32(FLEXSPI2_BASE + 0x100u)
#define FS2_LUT(n)      REG32(FLEXSPI2_BASE + 0x200u + 4u * (n))

#define MCR0_AHBGRANTWAIT(n) (((uint32_t)(n) & 0xFFu) << 24)
#define MCR0_IPGRANTWAIT(n)  (((uint32_t)(n) & 0xFFu) << 16)
#define MCR0_SCKFREERUNEN    (1u << 14)
#define MCR0_COMBINATIONEN   (1u << 13)
#define MCR0_DOZEEN          (1u << 12)
#define MCR0_HSEN            (1u << 11)
#define MCR0_ATDFEN          (1u << 7)
#define MCR0_ARDFEN          (1u << 6)
#define MCR0_RXCLKSRC(n)     (((uint32_t)(n) & 0x03u) << 4)
#define MCR0_MDIS            (1u << 1)
#define MCR0_SWRESET         (1u << 0)
#define MCR1_SEQWAIT(n)      (((uint32_t)(n) & 0xFFFFu) << 16)
#define MCR1_AHBBUSWAIT(n)   (((uint32_t)(n) & 0xFFFFu) << 0)
#define MCR2_RESUMEWAIT(n)   (((uint32_t)(n) & 0xFFu) << 24)
#define AHBCR_READADDROPT    (1u << 6)
#define AHBCR_PREFETCHEN     (1u << 5)
#define AHBCR_BUFFERABLEEN   (1u << 4)
#define AHBCR_CACHABLEEN     (1u << 3)
#define RXBUFCR0_PREFETCHEN  (1u << 31)
#define RXBUFCR0_BUFSZ(n)    (((uint32_t)(n) & 0xFFu) << 0)
#define FLSHCR1_TCSH(n)      (((uint32_t)(n) & 0x1Fu) << 5)
#define FLSHCR1_TCSS(n)      (((uint32_t)(n) & 0x1Fu) << 0)
#define FLSHCR2_AWRSEQID(n)  (((uint32_t)(n) & 0x0Fu) << 8)
#define FLSHCR2_ARDSEQID(n)  (((uint32_t)(n) & 0x0Fu) << 0)
#define IPCR1_ISEQID(n)      (((uint32_t)(n) & 0x0Fu) << 16)
#define IPCR1_IDATSZ(n)      (((uint32_t)(n) & 0xFFFFu) << 0)
#define IPCMD_TRG            (1u << 0)
#define INTR_IPCMDDONE       (1u << 0)
#define INTR_IPRXWA          (1u << 5)
#define LUTKEY_VALUE         0x5AF05AF0u
#define LUTCR_UNLOCK         (1u << 1)

/* LUT instruction: OPCODE[15:10] | NUM_PADS[9:8] | OPERAND[7:0]. Two
 * instructions pack into each 32-bit LUT word. */
#define LUT_INSTR(op, pads, oper) \
	((uint32_t)((((op) & 0x3Fu) << 10) | (((pads) & 0x3u) << 8) | ((oper) & 0xFFu)))
#define LUT_LO(op, pads, oper) LUT_INSTR(op, pads, oper)
#define LUT_HI(op, pads, oper) (LUT_INSTR(op, pads, oper) << 16)

#define OP_CMD   0x01u
#define OP_RADDR 0x02u
#define OP_WRITE 0x08u
#define OP_READ  0x09u
#define OP_DUMMY 0x0Cu
#define PADS1    0x00u
#define PADS4    0x02u

/* LUT sequence slots (each sequence is 4 LUT words apart). */
#define SEQ_EXIT_QPI   0
#define SEQ_RST_ENABLE 1
#define SEQ_RESET      2
#define SEQ_READ_ID    3
#define SEQ_ENTER_QPI  4
#define SEQ_READ_QPI   5
#define SEQ_WRITE_QPI  6

/* ------------------------------------------------------------ clock table */

struct clk_entry {
	uint8_t podf;
	uint8_t sel;
};

/* CCM_CBCMR FLEXSPI2_PODF / FLEXSPI2_CLK_SEL combinations. The comments are
 * PJRC's nominal labels; the REAL frequency of the PFD-sourced rows depends
 * on how the running OS programmed the fractions, so it is computed at
 * runtime — under Zephyr, PLL3 PFD0 sits near 262 MHz, not PJRC's 664.6,
 * which made "166.2 MHz" actually run (and bench) at ~65. */
static const struct clk_entry clk_table[PSRAM_CLK_COUNT] = {
	{ 5, 3 },   /*  88.0 with PLL2 (fixed, trustworthy)   */
	{ 3, 0 },   /*  99.0 with PJRC's PLL2 PFD2 of 396     */
	{ 6, 1 },   /* 102.9 with PLL3 PFD1 of 720            */
	{ 4, 3 },   /* 105.6 with PLL2                        */
	{ 5, 2 },   /* 110.8 with PJRC's PLL3 PFD0 of 664.6   */
	{ 5, 1 },   /* 120.0 with PLL3 PFD1 of 720            */
	{ 3, 3 },   /* 132.0 with PLL2                        */
	{ 4, 1 },   /* 144.0 with PLL3 PFD1 of 720            */
	{ 3, 2 },   /* 166.2 with PJRC's PLL3 PFD0 of 664.6   */
	{ 2, 3 },   /* 176.0 with PLL2                        */
};

static uint32_t fs2_source_hz(uint8_t sel)
{
	uint32_t frac;

	switch (sel) {
	case 0u:   /* PLL2 PFD2 */
		frac = (CCMA_PFD_528 >> 16) & 0x3Fu;
		break;
	case 1u:   /* PLL3 PFD0 (RM 14.7.6: CBCMR[FLEXSPI2_CLK_SEL]=01) */
		frac = CCMA_PFD_480 & 0x3Fu;
		break;
	case 2u:   /* PLL3 PFD1 (CBCMR[FLEXSPI2_CLK_SEL]=10) */
		frac = (CCMA_PFD_480 >> 8) & 0x3Fu;
		break;
	default:   /* PLL2, fixed */
		return 528000000u;
	}
	if (frac == 0u) {
		return 0u;
	}
	return (uint32_t)(((uint64_t)(sel == 0u ? 528000000u : 480000000u) * 18u) / frac);
}

uint32_t psram_clock_hz(int index)
{
	if (index < 0 || index >= PSRAM_CLK_COUNT) {
		return 0u;
	}
	return fs2_source_hz(clk_table[index].sel) / ((uint32_t)clk_table[index].podf + 1u);
}

static struct psram_info g_info;

const struct psram_info *psram_info(void)
{
	return &g_info;
}

/* ------------------------------------------------------------- low level */

static void fs2_command(uint32_t seq, uint32_t addr)
{
	FS2_IPCR0 = addr;
	FS2_IPCR1 = IPCR1_ISEQID(seq);
	FS2_IPCMD = IPCMD_TRG;
	while (!(FS2_INTR & INTR_IPCMDDONE)) {
	}
	FS2_INTR = INTR_IPCMDDONE;
}

static uint32_t fs2_read_id(uint32_t addr)
{
	uint32_t id;

	FS2_IPCR0 = addr;
	FS2_IPCR1 = IPCR1_ISEQID(SEQ_READ_ID) | IPCR1_IDATSZ(4);
	FS2_IPCMD = IPCMD_TRG;
	while (!(FS2_INTR & INTR_IPCMDDONE)) {
	}
	id = FS2_RFDR0;
	FS2_INTR = INTR_IPCMDDONE | INTR_IPRXWA;
	return id;
}

/*
 * Identify the part at `addr`. Returns capacity in MB, 0 if nothing answers.
 *
 * The chip may still be in QPI mode from a previous run (a warm reset does not
 * reset the PSRAM), so exit QPI and reset before asking for the ID — otherwise
 * a perfectly good part reads back as absent after the first reboot.
 */
static uint8_t fs2_probe(uint32_t addr, uint32_t *raw_id)
{
	uint32_t id;

	fs2_command(SEQ_EXIT_QPI, addr);
	fs2_command(SEQ_RST_ENABLE, addr);
	fs2_command(SEQ_RESET, addr);
	id = fs2_read_id(addr);
	if (raw_id != NULL) {
		*raw_id = id;
	}

	switch (id & 0xFFFFu) {
	case 0x5D0Du:            /* AP Memory / Ipus / ESP / Lyontek, 8 MB */
		return 8u;
	case 0x5D9Du:            /* ISSI: size lives in ID bits 23:21     */
		switch ((id >> 21) & 0x7u) {
		case 0x3u:
			return 8u;
		case 0x4u:
			return 16u;   /* IS66WVS16M8 — two 8 MB dies in one package */
		default:
			return 0u;
		}
	default:
		return 0u;
	}
}

static void fs2_pins(void)
{
	static const uint32_t pad_cfg[8] = {
		0x1B0F9u, /* EMC_22 A_SS1_B  — 100K pullup, strong drive, hyst */
		0x110F9u, /* EMC_23 A_DQS    — keeper                          */
		0x1B0F9u, /* EMC_24 A_SS0_B  — 100K pullup                     */
		0x100F9u, /* EMC_25 A_SCLK                                     */
		0x170F9u, /* EMC_26 A_DATA0  — 47K pullup                      */
		0x170F9u, /* EMC_27 A_DATA1                                    */
		0x170F9u, /* EMC_28 A_DATA2                                    */
		0x170F9u, /* EMC_29 A_DATA3                                    */
	};
	int i;

	for (i = 0; i < 8; i++) {
		REG32(IOMUXC_PAD_EMC22 + 4u * (uint32_t)i) = pad_cfg[i];
		/* ALT8 selects FLEXSPI2_A_*; bit 4 is SION, which the DQS and
		 * data lines need so the controller can read the pad back. */
		REG32(IOMUXC_MUX_EMC22 + 4u * (uint32_t)i) = 8u | 0x10u;
	}

	/* Daisy-chain selects: route FlexSPI2 port A to the EMC_2x pads. */
	IOMUXC_SEL_DQS_FA = 1u;
	IOMUXC_SEL_IO_FA0 = 1u;
	IOMUXC_SEL_IO_FA1 = 1u;
	IOMUXC_SEL_IO_FA2 = 1u;
	IOMUXC_SEL_IO_FA3 = 1u;
	IOMUXC_SEL_SCK_FA = 1u;
}

static void fs2_clock(int index)
{
	const struct clk_entry *c = &clk_table[index];

	CCM_CBCMR = (CCM_CBCMR & ~((0x7u << 29) | (0x3u << 8))) |
		    ((uint32_t)c->podf << 29) | ((uint32_t)c->sel << 8);
	CCM_CCGR7 |= (3u << 2);   /* CCGR7[FLEXSPI2] = always on */
}

static void fs2_controller(void)
{
	FS2_MCR0 |= MCR0_MDIS;
	/* Clear the same set PJRC does — critically ATDFEN and ARDFEN, which
	 * RESET TO 1 (MCR0 reset value ends 0xC2) and route the IP TX/RX
	 * FIFOs to DMA. With ARDFEN left set, RFDR reads return zeros while
	 * IPCMDDONE still completes, so every chip "answers" id 0x00000000
	 * and a perfectly soldered board probes as empty. */
	FS2_MCR0 = (FS2_MCR0 & ~(MCR0_AHBGRANTWAIT(0xFF) | MCR0_IPGRANTWAIT(0xFF) |
				 MCR0_SCKFREERUNEN | MCR0_COMBINATIONEN |
				 MCR0_DOZEEN | MCR0_HSEN | MCR0_ATDFEN |
				 MCR0_ARDFEN | MCR0_RXCLKSRC(3) | MCR0_SWRESET)) |
		   MCR0_AHBGRANTWAIT(0xFF) | MCR0_IPGRANTWAIT(0xFF) |
		   MCR0_RXCLKSRC(1) | MCR0_MDIS;
	FS2_MCR1 = MCR1_SEQWAIT(0xFFFF) | MCR1_AHBBUSWAIT(0xFFFF);
	FS2_MCR2 = (FS2_MCR2 & ~MCR2_RESUMEWAIT(0xFF)) | MCR2_RESUMEWAIT(0x20);

	/*
	 * AHB read path. This is the tuning knob that matters most for token
	 * rate: weight streaming is one long sequential burst per matrix row,
	 * so the prefetcher gets to work ahead nearly perfectly. Buffers 0 and
	 * 1 are given the full 512 B (BUFSZ counts 64-bit words) with prefetch
	 * on; 2 and 3 are left minimal because nothing else is competing for
	 * the bus during inference.
	 */
	FS2_AHBCR &= ~(AHBCR_READADDROPT | AHBCR_PREFETCHEN |
		       AHBCR_BUFFERABLEEN | AHBCR_CACHABLEEN);
	FS2_AHBRXBUFCR0(0) = RXBUFCR0_PREFETCHEN | RXBUFCR0_BUFSZ(64);
	FS2_AHBRXBUFCR0(1) = RXBUFCR0_PREFETCHEN | RXBUFCR0_BUFSZ(64);
	FS2_AHBRXBUFCR0(2) = 0u;
	FS2_AHBRXBUFCR0(3) = 0u;

	FS2_IPRXFCR = (FS2_IPRXFCR & 0xFFFFFFC0u) | 1u;
	FS2_IPTXFCR = (FS2_IPTXFCR & 0xFFFFFFC0u) | 1u;
	FS2_INTEN = 0u;

	FS2_FLSHA1CR1 = FLSHCR1_TCSH(1) | FLSHCR1_TCSS(1);
	FS2_FLSHA2CR1 = FLSHCR1_TCSH(1) | FLSHCR1_TCSS(1);
	FS2_FLSHA1CR2 = FLSHCR2_AWRSEQID(SEQ_WRITE_QPI) |
			FLSHCR2_ARDSEQID(SEQ_READ_QPI);
	FS2_FLSHA2CR2 = FLSHCR2_AWRSEQID(SEQ_WRITE_QPI) |
			FLSHCR2_ARDSEQID(SEQ_READ_QPI);

	FS2_MCR0 &= ~MCR0_MDIS;
}

static void fs2_lut(void)
{
	int i;

	FS2_LUTKEY = LUTKEY_VALUE;
	FS2_LUTCR = LUTCR_UNLOCK;
	for (i = 0; i < 64; i++) {
		FS2_LUT(i) = 0u;
	}
	FS2_MCR0 |= MCR0_SWRESET;
	while (FS2_MCR0 & MCR0_SWRESET) {
	}

	FS2_LUTKEY = LUTKEY_VALUE;
	FS2_LUTCR = LUTCR_UNLOCK;

	/* 0xF5 exit QPI (issued in quad mode, in case we are already there) */
	FS2_LUT(SEQ_EXIT_QPI * 4) = LUT_LO(OP_CMD, PADS4, 0xF5u);
	/* 0x66 / 0x99 reset-enable then reset, single lane */
	FS2_LUT(SEQ_RST_ENABLE * 4) = LUT_LO(OP_CMD, PADS1, 0x66u);
	FS2_LUT(SEQ_RESET * 4) = LUT_LO(OP_CMD, PADS1, 0x99u);
	/* 0x9F read ID: 24 dummy cycles then 1 byte, single lane */
	FS2_LUT(SEQ_READ_ID * 4 + 0) = LUT_LO(OP_CMD, PADS1, 0x9Fu) |
				       LUT_HI(OP_DUMMY, PADS1, 24u);
	FS2_LUT(SEQ_READ_ID * 4 + 1) = LUT_LO(OP_READ, PADS1, 1u);
	/* 0x35 enter QPI */
	FS2_LUT(SEQ_ENTER_QPI * 4) = LUT_LO(OP_CMD, PADS1, 0x35u);
	/* 0xEB quad fast read, 24-bit address, 6 dummy cycles */
	FS2_LUT(SEQ_READ_QPI * 4 + 0) = LUT_LO(OP_CMD, PADS4, 0xEBu) |
					LUT_HI(OP_RADDR, PADS4, 24u);
	FS2_LUT(SEQ_READ_QPI * 4 + 1) = LUT_LO(OP_DUMMY, PADS4, 6u) |
					LUT_HI(OP_READ, PADS4, 1u);
	/* 0x38 quad write, 24-bit address */
	FS2_LUT(SEQ_WRITE_QPI * 4 + 0) = LUT_LO(OP_CMD, PADS4, 0x38u) |
					 LUT_HI(OP_RADDR, PADS4, 24u);
	FS2_LUT(SEQ_WRITE_QPI * 4 + 1) = LUT_LO(OP_WRITE, PADS4, 1u);
}

/* ------------------------------------------------------------------- init */

int psram_init(int clock_index)
{
	uint8_t size1, size2;
	uint32_t id1 = 0, id2 = 0;

	if (clock_index < 0 || clock_index >= PSRAM_CLK_COUNT) {
		return -EINVAL;
	}

	memset(&g_info, 0, sizeof(g_info));
	g_info.clock_index = (uint8_t)clock_index;
	g_info.clock_hz = psram_clock_hz(clock_index);

	fs2_pins();
	fs2_clock(clock_index);
	fs2_controller();
	fs2_lut();

	/* Chip 1 sits on A_SS0_B (pad EMC_24, the footprint PJRC labels for
	 * RAM). Chip 2 is on A_SS1_B (EMC_22, the footprint that can also take
	 * a NOR flash). FLSHA*CR0 is the size in KB, so `mb << 10`. */
	size1 = fs2_probe(0u, &id1);
	if (size1 == 0u) {
		LOG_WRN("no PSRAM on CS0 (id 0x%08x) — nothing soldered, or "
			"a solder bridge / cold joint on the QSPI pads", id1);
		return 0;
	}
	FS2_FLSHA1CR0 = (uint32_t)size1 << 10;
	fs2_command(SEQ_ENTER_QPI, 0u);

	size2 = fs2_probe((uint32_t)size1 << 20, &id2);
	if (size2 > 0u) {
		FS2_FLSHA2CR0 = (uint32_t)size2 << 10;
		fs2_command(SEQ_ENTER_QPI, (uint32_t)size1 << 20);
	} else {
		FS2_FLSHA2CR0 = 0u;
	}

	g_info.chips = (uint8_t)(1u + (size2 > 0u));
	g_info.chip_mbytes[0] = size1;
	g_info.chip_mbytes[1] = size2;
	g_info.chip_id[0] = id1;
	g_info.chip_id[1] = id2;
	g_info.mbytes = (uint32_t)size1 + size2;

	barrier_dsync_fence_full();
	LOG_INF("PSRAM %u MB (%u chip%s) @ %u.%u MHz",
		g_info.mbytes, g_info.chips, g_info.chips == 1 ? "" : "s",
		g_info.clock_hz / 1000000u, (g_info.clock_hz / 100000u) % 10u);
	return (int)g_info.mbytes;
}

/* -------------------------------------------------------------- benchmarks */

/*
 * Read the region with 32-bit loads and report KB/s.
 *
 * Deliberately a plain load loop rather than memcpy: this is the access
 * pattern the matvec kernel actually generates (sequential, read-once, no
 * reuse), so the number it produces is the number that predicts token rate.
 * `volatile` on the accumulator keeps the optimiser from deleting the loads.
 */
uint32_t psram_bench_read(uint32_t off, size_t bytes)
{
	volatile uint32_t sink = 0;
	const uint32_t *p = (const uint32_t *)(uintptr_t)(PSRAM_BASE + off);
	size_t words = bytes / 4u;
	uint32_t acc = 0;
	int64_t t0, dt_us;
	size_t i;

	t0 = k_uptime_ticks();
	for (i = 0; i < words; i += 8) {
		acc += p[i] + p[i + 1] + p[i + 2] + p[i + 3];
		acc += p[i + 4] + p[i + 5] + p[i + 6] + p[i + 7];
	}
	dt_us = k_ticks_to_us_floor64(k_uptime_ticks() - t0);
	sink = acc;
	(void)sink;

	if (dt_us <= 0) {
		return 0u;
	}
	return (uint32_t)(((uint64_t)bytes * 1000u) / (uint64_t)dt_us);
}

uint32_t psram_bench_write(uint32_t off, size_t bytes)
{
	uint32_t *p = (uint32_t *)(uintptr_t)(PSRAM_BASE + off);
	size_t words = bytes / 4u;
	int64_t t0, dt_us;
	size_t i;

	t0 = k_uptime_ticks();
	for (i = 0; i < words; i += 8) {
		p[i] = (uint32_t)i;
		p[i + 1] = (uint32_t)i + 1u;
		p[i + 2] = (uint32_t)i + 2u;
		p[i + 3] = (uint32_t)i + 3u;
		p[i + 4] = (uint32_t)i + 4u;
		p[i + 5] = (uint32_t)i + 5u;
		p[i + 6] = (uint32_t)i + 6u;
		p[i + 7] = (uint32_t)i + 7u;
	}
	barrier_dsync_fence_full();
	dt_us = k_ticks_to_us_floor64(k_uptime_ticks() - t0);

	if (dt_us <= 0) {
		return 0u;
	}
	return (uint32_t)(((uint64_t)bytes * 1000u) / (uint64_t)dt_us);
}

/* ---------------------------------------------------------------- memtest */

uint32_t psram_memtest(uint32_t off, size_t bytes, bool quick)
{
	volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)(PSRAM_BASE + off);
	size_t words = bytes / 4u;
	size_t stride = quick ? (4096u / 4u) : 1u;
	uint32_t bad = 0;
	size_t i;

	/*
	 * Address-in-address, then its complement. This catches the two
	 * failure modes that actually happen on a hand-soldered QSPI part:
	 * a stuck or shorted data line (complement pass fails) and a
	 * mis-decoded address line (a word reads back another word's value,
	 * which address-in-address makes obvious).
	 */
	for (i = 0; i < words; i += stride) {
		p[i] = (uint32_t)i ^ 0xA5A5A5A5u;
	}
	barrier_dsync_fence_full();
	for (i = 0; i < words; i += stride) {
		if (p[i] != ((uint32_t)i ^ 0xA5A5A5A5u)) {
			bad++;
		}
	}
	for (i = 0; i < words; i += stride) {
		p[i] = ~((uint32_t)i ^ 0xA5A5A5A5u);
	}
	barrier_dsync_fence_full();
	for (i = 0; i < words; i += stride) {
		if (p[i] != ~((uint32_t)i ^ 0xA5A5A5A5u)) {
			bad++;
		}
	}
	return bad;
}

/* ------------------------------------------------------------ clock sweep */

int psram_clock_sweep(int max_index)
{
	int order[PSRAM_CLK_COUNT];
	uint32_t best_mb = 0u;
	int best = -1;
	int i, j, n = 0;

	if (max_index < 0 || max_index >= PSRAM_CLK_COUNT) {
		max_index = PSRAM_CLK_COUNT - 1;
	}

	/* Walk the clocks in ascending REAL frequency. The table's nominal
	 * order is only sorted under PJRC's PFD programming; under Zephyr's
	 * it is not even monotonic. */
	for (i = 0; i <= max_index; i++) {
		order[n++] = i;
	}
	for (i = 1; i < n; i++) {
		for (j = i; j > 0 && psram_clock_hz(order[j]) <
				     psram_clock_hz(order[j - 1]); j--) {
			int t = order[j];

			order[j] = order[j - 1];
			order[j - 1] = t;
		}
	}

	for (i = 0; i < n; i++) {
		int idx = order[i];
		uint32_t hz = psram_clock_hz(idx);
		uint32_t mb, bad, kbps;
		size_t test_bytes;

		mb = (uint32_t)psram_init(idx);
		if (mb == 0u || mb < best_mb) {
			/* Enumerating fewer chips than a slower clock did is
			 * a failure, not a smaller success — losing CS1 at
			 * speed must not count as "pass". */
			LOG_WRN("%3u.%u MHz: %s", hz / 1000000u,
				(hz / 100000u) % 10u,
				mb == 0u ? "chip did not enumerate"
					 : "capacity shrank — not stable");
			if (best >= 0) {
				break;
			}
			continue;
		}

		/* Test a 1 MB window near the top of the array — the far end of
		 * the address bus is where marginal timing shows up first. */
		test_bytes = 1024u * 1024u;
		bad = psram_memtest((mb << 20) - (uint32_t)test_bytes,
				    test_bytes, false);
		kbps = psram_bench_read(0u, 512u * 1024u);

		LOG_INF("%3u.%u MHz: %s  read %u.%u MB/s",
			hz / 1000000u, (hz / 100000u) % 10u,
			bad ? "FAIL" : "pass", kbps / 1000u, (kbps / 100u) % 10u);

		if (bad == 0u) {
			best = idx;
			best_mb = mb;
		} else if (best >= 0) {
			/* Once it starts failing it will keep failing; stop
			 * hammering a part that is already out of spec. */
			break;
		}
	}

	if (best < 0) {
		LOG_ERR("no usable FlexSPI2 clock found");
		return -EIO;
	}
	psram_init(best);
	LOG_INF("settled on %u.%u MHz",
		psram_clock_hz(best) / 1000000u,
		(psram_clock_hz(best) / 100000u) % 10u);
	return best;
}
