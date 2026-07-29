/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 *
 * USB CDC-ACM console bring-up for the USBD "next" stack.
 *
 * The chosen zephyr,console / zephyr,shell-uart nodes point at cdc_acm_uart0
 * (boards/teensy41.overlay), but unlike the legacy stack, the next stack
 * instantiates nothing on its own: the application must define the device
 * context, attach descriptors and configurations, and enable it. This file
 * is that boilerplate, run once at APPLICATION init so the console is live
 * before main() starts. SHELL_BACKEND_SERIAL_CHECK_DTR=n in prj.conf keeps
 * boot from blocking when no host terminal is attached.
 */

#include <zephyr/device.h>
#include <zephyr/init.h>
#include <zephyr/logging/log.h>
#include <zephyr/usb/usbd.h>

LOG_MODULE_REGISTER(usb_console, CONFIG_TINYLLM_LOG_LEVEL);

/* pid.codes test allocation — the Zephyr project VID is reserved for
 * upstream samples, and squatting PJRC's would misidentify the board. */
#define TINYLLM_USB_VID 0x1209
#define TINYLLM_USB_PID 0x0001

USBD_DEVICE_DEFINE(tinyllm_usbd, DEVICE_DT_GET(DT_NODELABEL(zephyr_udc0)),
		   TINYLLM_USB_VID, TINYLLM_USB_PID);

USBD_DESC_LANG_DEFINE(tinyllm_lang);
USBD_DESC_MANUFACTURER_DEFINE(tinyllm_mfr, "teensytinyllm");
USBD_DESC_PRODUCT_DEFINE(tinyllm_product, "Teensy 4.1 tinyllm console");
USBD_DESC_SERIAL_NUMBER_DEFINE(tinyllm_sn);
USBD_DESC_CONFIG_DEFINE(tinyllm_fs_desc, "FS Configuration");
USBD_DESC_CONFIG_DEFINE(tinyllm_hs_desc, "HS Configuration");

/* Bus powered; bMaxPower 250 × 2 mA = 500 mA covers the board plus SD and
 * both PSRAM chips with margin. */
USBD_CONFIGURATION_DEFINE(tinyllm_fs_config, 0, 250, &tinyllm_fs_desc);
USBD_CONFIGURATION_DEFINE(tinyllm_hs_config, 0, 250, &tinyllm_hs_desc);

static int add_speed(enum usbd_speed speed, struct usbd_config_node *cfg)
{
	int err;

	err = usbd_add_configuration(&tinyllm_usbd, speed, cfg);
	if (err) {
		return err;
	}
	err = usbd_register_all_classes(&tinyllm_usbd, speed, 1);
	if (err) {
		return err;
	}
	/* CDC-ACM is a multi-interface function: the IAD code triple tells
	 * the host to read class info per interface. */
	usbd_device_set_code_triple(&tinyllm_usbd, speed,
				    USB_BCC_MISCELLANEOUS, 0x02, 0x01);
	return 0;
}

static int usb_console_enable(void)
{
	struct usbd_desc_node *descs[] = {
		&tinyllm_lang, &tinyllm_mfr, &tinyllm_product, &tinyllm_sn,
	};
	unsigned int i;
	int err;

	for (i = 0; i < ARRAY_SIZE(descs); i++) {
		err = usbd_add_descriptor(&tinyllm_usbd, descs[i]);
		if (err) {
			LOG_ERR("descriptor %u: %d", i, err);
			return err;
		}
	}

	/* The RT1062's EHCI controller is high-speed; register both speeds
	 * so a FS-only hub still gets a working console. */
	if (usbd_caps_speed(&tinyllm_usbd) == USBD_SPEED_HS) {
		err = add_speed(USBD_SPEED_HS, &tinyllm_hs_config);
		if (err) {
			LOG_ERR("HS config: %d", err);
			return err;
		}
	}
	err = add_speed(USBD_SPEED_FS, &tinyllm_fs_config);
	if (err) {
		LOG_ERR("FS config: %d", err);
		return err;
	}

	err = usbd_init(&tinyllm_usbd);
	if (err) {
		LOG_ERR("usbd_init: %d", err);
		return err;
	}
	err = usbd_enable(&tinyllm_usbd);
	if (err) {
		LOG_ERR("usbd_enable: %d", err);
		return err;
	}
	return 0;
}

SYS_INIT(usb_console_enable, APPLICATION, CONFIG_APPLICATION_INIT_PRIORITY);
