/*
 * Copyright (c) 2026 Ian Adelman
 * SPDX-License-Identifier: MIT
 */

#include <string.h>

#include <zephyr/fs/fs.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/storage/disk_access.h>
#include <ff.h>

#include "model_store.h"
#include "psram_flexspi2.h"

LOG_MODULE_REGISTER(model_store, CONFIG_TINYLLM_LOG_LEVEL);

#define SD_MOUNT "/SD:"

/* Everything the USDHC ADMA engine writes must be non-cacheable: the v4.0
 * driver does no cache maintenance on data buffers, so a cacheable target
 * serves stale lines to the CPU afterwards (this corrupted the first real
 * model image in transit). That covers the staging buffer below and FATFS
 * itself — its embedded sector window is the DMA target for all metadata
 * and small/unaligned reads. */
static FATFS fat_fs __nocache;
static struct fs_mount_t mp = {
	.type = FS_FATFS,
	.fs_data = &fat_fs,
	.mnt_point = SD_MOUNT,
};
static bool mounted;

/*
 * 32 KB. SDIO throughput on the Teensy's slot collapses below about 16 KB per
 * read (per-block command overhead dominates) and stops improving much above
 * 32 KB. This buffer is also the streaming tile, so it has to be at least one
 * full matrix row: 32 KB covers a 4096-wide Q8_0 row with room to spare.
 */
#define STAGE_BYTES (32u * 1024u)
static uint8_t stage_buf[STAGE_BYTES] __nocache __aligned(32);

int model_sd_mount(void)
{
	static const char *disk = "SD";

	/* The .nocache section is NOLOAD — never zeroed at boot — so the
	 * FATFS struct starts as stack garbage unless cleared here. */
	if (!mounted) {
		memset(&fat_fs, 0, sizeof(fat_fs));
	}
	uint32_t block_count = 0, block_size = 0;
	int rc;

	if (mounted) {
		return 0;
	}

	rc = disk_access_init(disk);
	if (rc != 0) {
		LOG_ERR("no SD card (disk_access_init=%d) — card seated? "
			"FAT32/exFAT formatted?", rc);
		return rc;
	}
	(void)disk_access_ioctl(disk, DISK_IOCTL_GET_SECTOR_COUNT, &block_count);
	(void)disk_access_ioctl(disk, DISK_IOCTL_GET_SECTOR_SIZE, &block_size);

	rc = fs_mount(&mp);
	if (rc != 0) {
		LOG_ERR("fs_mount: %d", rc);
		return rc;
	}
	mounted = true;
	LOG_INF("SD mounted at %s (%u MB)", SD_MOUNT,
		(unsigned)(((uint64_t)block_count * block_size) >> 20));
	return 0;
}

static int open_sized(const char *path, struct fs_file_t *f, uint64_t *size)
{
	struct fs_dirent ent;
	int rc;

	rc = fs_stat(path, &ent);
	if (rc != 0) {
		LOG_ERR("stat %s: %d", path, rc);
		return rc;
	}
	*size = ent.size;

	fs_file_t_init(f);
	rc = fs_open(f, path, FS_O_READ);
	if (rc != 0) {
		LOG_ERR("open %s: %d", path, rc);
		return rc;
	}
	return 0;
}

int model_load_to_psram(const char *path, uint32_t psram_off, TqStore *store,
			uint64_t *out_bytes,
			void (*progress)(uint64_t done, uint64_t total))
{
	struct fs_file_t f;
	uint64_t size = 0, done = 0;
	uint8_t *dst;
	const struct psram_info *pi = psram_info();
	int rc;

	rc = model_sd_mount();
	if (rc != 0) {
		return rc;
	}
	rc = open_sized(path, &f, &size);
	if (rc != 0) {
		return rc;
	}

	if (pi->mbytes == 0u) {
		LOG_ERR("no PSRAM detected — use `tinyllm stream` instead");
		fs_close(&f);
		return -ENODEV;
	}
	if (psram_off + size > ((uint64_t)pi->mbytes << 20)) {
		LOG_ERR("model is %llu MB but only %u MB of PSRAM is fitted; "
			"stream it from the card instead",
			(unsigned long long)(size >> 20), pi->mbytes);
		fs_close(&f);
		return -ENOSPC;
	}

	dst = (uint8_t *)(uintptr_t)(PSRAM_BASE + psram_off);
	while (done < size) {
		uint32_t want = (uint32_t)MIN((uint64_t)STAGE_BYTES, size - done);
		ssize_t got = fs_read(&f, stage_buf, want);

		if (got <= 0) {
			LOG_ERR("read at %llu: %d", (unsigned long long)done, (int)got);
			fs_close(&f);
			return (got == 0) ? -EIO : (int)got;
		}
		memcpy(dst + done, stage_buf, (size_t)got);
		done += (uint64_t)got;
		if (progress != NULL) {
			progress(done, size);
		}
	}
	fs_close(&f);

	tq_store_init_mapped(store, dst, size);
	if (out_bytes != NULL) {
		*out_bytes = size;
	}
	LOG_INF("loaded %s: %llu bytes into PSRAM+0x%x",
		path, (unsigned long long)size, psram_off);
	return 0;
}

/* --------------------------------------------------------------- streaming */

static struct fs_file_t stream_file;
static bool stream_open;
static struct k_mutex stream_lock;

static int stream_read(void *ctx, uint64_t off, void *dst, uint32_t nbytes)
{
	uint32_t done = 0;
	int rc = 0;

	ARG_UNUSED(ctx);
	if (!stream_open) {
		return -EBADF;
	}

	k_mutex_lock(&stream_lock, K_FOREVER);
	if (fs_seek(&stream_file, (off_t)off, FS_SEEK_SET) != 0) {
		rc = -EIO;
	}
	/* Bounce through the non-cacheable staging buffer: the engine hands
	 * this callback arbitrary (cacheable) destinations, which must never
	 * be DMA targets — see the note above fat_fs. */
	while (rc == 0 && done < nbytes) {
		uint32_t want = MIN(STAGE_BYTES, nbytes - done);
		ssize_t got = fs_read(&stream_file, stage_buf, want);

		if (got <= 0) {
			rc = -EIO;
			break;
		}
		memcpy((uint8_t *)dst + done, stage_buf, (size_t)got);
		done += (uint32_t)got;
	}
	k_mutex_unlock(&stream_lock);
	return rc;
}

int model_open_streaming(const char *path, TqStore *store, uint8_t *tile,
			 uint32_t tile_bytes, uint64_t *out_bytes)
{
	uint64_t size = 0;
	int rc;

	rc = model_sd_mount();
	if (rc != 0) {
		return rc;
	}
	model_close_streaming();
	k_mutex_init(&stream_lock);

	rc = open_sized(path, &stream_file, &size);
	if (rc != 0) {
		return rc;
	}
	stream_open = true;

	tq_store_init_stream(store, size, stream_read, NULL, tile, tile_bytes);
	if (out_bytes != NULL) {
		*out_bytes = size;
	}
	LOG_INF("streaming %s: %llu bytes, %u KB tile",
		path, (unsigned long long)size, tile_bytes / 1024u);
	LOG_WRN("streaming mode: the tokenizer needs random access and is "
		"unavailable; generation emits raw token ids");
	return 0;
}

void model_close_streaming(void)
{
	if (stream_open) {
		fs_close(&stream_file);
		stream_open = false;
	}
}
