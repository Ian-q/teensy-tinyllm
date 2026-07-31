/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * PJRC soft-reboot convention: a host that sets the CDC baud rate to 134
 * is asking the firmware to jump into the HalfKay bootloader, exactly as
 * Teensyduino cores do (usb.c checks dwDTERate == 134). `bkpt #251` is
 * trapped by the Teensy's bootloader chip. This makes reflashing
 * hands-free:
 *
 *     stty -f /dev/cu.usbmodemXXXX 134
 *     teensy_loader_cli --mcu=TEENSY41 -w zephyr.hex
 *
 * The secure-boot ROM variant of Teensyduino's _reboot_Teensyduino_() is
 * deliberately not ported: it applies only to locked parts, which no
 * stock Teensy is.
 */

#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/init.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(usb_reboot, CONFIG_TINYLLM_LOG_LEVEL);

#define REBOOT_BAUD 134u
#define POLL_MS     100

static const struct device *const console_dev =
	DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

static void reboot_poll(struct k_work *work);
static K_WORK_DELAYABLE_DEFINE(reboot_work, reboot_poll);

static void reboot_poll(struct k_work *work)
{
	uint32_t baud = 0;

	ARG_UNUSED(work);

	if (uart_line_ctrl_get(console_dev, UART_LINE_CTRL_BAUD_RATE,
			       &baud) == 0 && baud == REBOOT_BAUD) {
		LOG_WRN("host set baud 134 — rebooting into HalfKay");
		/* Give the control transfer and log a moment to finish;
		 * blocking the work queue is fine, nothing survives this. */
		k_msleep(50);
		__disable_irq();
		__asm__ volatile("bkpt #251");
	}

	k_work_schedule(&reboot_work, K_MSEC(POLL_MS));
}

static int usb_reboot_init(void)
{
	k_work_schedule(&reboot_work, K_MSEC(POLL_MS));
	return 0;
}

SYS_INIT(usb_reboot_init, APPLICATION, 99);
