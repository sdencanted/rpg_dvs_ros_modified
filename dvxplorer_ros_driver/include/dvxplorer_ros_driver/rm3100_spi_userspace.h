/*
 * RM3100 User-space driver (using spidev driver)
 *
 * Copyright (c) 2007  MontaVista Software, Inc.
 * Copyright (c) 2007  Anton Vorontsov <avorontsov@ru.mvista.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License.
 *
 */

#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <sys/stat.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>
#include <gpiod.h>
#include <sys/time.h>

#ifndef	CONSUMER
#define	CONSUMER	"Consumer"
#endif

#define POLL_GPIO POLLPRI | POLLERR

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

//internal register values without the R/W bit
#define RM3100_REVID_REG 0x36 // Hexadecimal address for the Revid internal register
#define RM3100_POLL_REG 0x00 // Hexadecimal address for the Poll internal register
#define RM3100_CMM_REG 0x01 // Hexadecimal address for the Continuous Measurement Mode internal register
#define RM3100_STATUS_REG 0x34 // Hexadecimal address for the Status internal register
#define RM3100_HSHAKE_REG 0x35 // Hexadecimal address for the Handshake register
#define RM3100_CCX1_REG 0x04 // Hexadecimal address for the Cycle Count X1 internal register
#define RM3100_CCX0_REG 0x05 // Hexadecimal address for the Cycle Count X0 internal register
#define RM3100_TMRC_REG 0x0B // Hexadecimal address for the TMRC internal register

 
#define SYSFS_GPIO_DIR "/sys/class/gpio"
#define MAX_BUF 64

struct MeasurementsUt{
	float x;
	float y;
	float z;
	int ret;
};
struct Measurements{
	long x;
	long y;
	long z;
	int ret;
};

void read_addr(int addr, int read_size,  uint8_t const *rx);
void write_addr(int addr, int write_size,  uint8_t const *tx);
int get_revid_id();

int get_measurement_ready();
int get_measurement_ready_drdy();
struct Measurements get_measurement_poll();

//Timeout in nanoseconds
struct Measurements get_measurement(int timeout,bool use_drdy);

struct MeasurementsUt get_measurement_ut(uint16_t cycleCount);
void set_continuous_measurement(bool enable);

void set_tmrc(uint8_t tmrc_value);
void poll_single_measurement();
struct Measurements single_measurement();
struct MeasurementsUt single_measurement_ut(uint16_t cycleCount);

//newCC is the new cycle count value (16 bits) to change the data acquisition
void change_cycle_count( uint16_t newCC);

int read_handshake();

int start_mag();
int end_mag();
// void clear_drdy();
int start_drdy();
int end_drdy();