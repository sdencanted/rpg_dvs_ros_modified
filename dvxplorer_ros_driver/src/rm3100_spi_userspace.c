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
 * Cross-compile with cross-gcc -I/path/to/cross-kernel/include
 */
#include "dvxplorer_ros_driver/rm3100_spi_userspace.h"

static void pabort(const char *s)
{
	perror(s);
	abort();
}

static const char *device = "/dev/spidev0.0";
static uint32_t mode;
static uint8_t bits = 8;
static uint32_t speed = 500000;
static uint16_t delay;
// static int verbose;
static int fd = 0;
struct gpiod_chip *chip;
struct gpiod_line *line;
struct gpiod_line_event event;
static char *chipname = "gpiochip0";
static unsigned int line_num = 85; // GPIO Pin #12 / PN.01

static void transfer(uint8_t const *tx, uint8_t const *rx, size_t len)
{
	int ret;
	// int out_fd;
	struct spi_ioc_transfer tr = {
		.tx_buf = (unsigned long)tx,
		.rx_buf = (unsigned long)rx,
		.len = len,
		.delay_usecs = delay,
		.speed_hz = speed,
		.bits_per_word = bits,
	};

	ret = ioctl(fd, SPI_IOC_MESSAGE(1), &tr);
	if (ret < 1)
		pabort("can't send spi message");
}

void read_addr(int addr, int read_size, uint8_t const *rx)
{
	uint8_t tx[1 + read_size];
	tx[0] = addr | 0x80;
	// for (int i = 1; i < read_size; ++i) {
	// 	tx[i] = 0x30;
	// }
	memset(&tx[1], 0x00, read_size * sizeof(uint8_t));
	transfer(tx, rx, read_size + 1);
	return;
}
void write_addr(int addr, int write_size, uint8_t const *tx)
{
	uint8_t tx_new[1 + write_size];
	tx_new[0] = addr & 0x7F;
	memcpy(&tx_new[1], &tx[0], write_size * sizeof(uint8_t));
	uint8_t rx[1 + write_size];
	transfer(tx_new, rx, write_size + 1);
	return;
}
int get_revid_id()
{
	uint8_t rx[2] = {
		0,
	};
	read_addr(RM3100_REVID_REG, 1, rx);
	return rx[1];
}

int get_measurement_ready()
{
	uint8_t rx[2] = {
		0,
	};
	read_addr(RM3100_STATUS_REG, 1, rx);
	return (rx[1] & 0x80) == 0x80;
}
int get_measurement_ready_drdy()
{
	int err = gpiod_line_get_value(line);
	if (err == -1)
		return end_drdy();
	return err;
}
int wait_drdy(int timeout)
{
	int err;
	struct timespec timeout_gpio;
	timeout_gpio.tv_nsec = timeout;
	timeout_gpio.tv_sec = 0;
	err = gpiod_line_event_wait(line, &timeout_gpio);
	if (err < 0)
	{
		perror("Wait failed");
		return end_drdy();
	}
	else if (err == 1)
	{
		err = gpiod_line_event_read(line, &event);
		if (err < 0)
		{
			perror("event read failed");
			return -1;
		}
		else
			return 1;
	}
	else
		return 0;
}

struct Measurements get_measurement_poll()
{
	return get_measurement(-1, false);
}
struct Measurements get_measurement(int timeout, bool use_drdy)
{

	// special bit manipulation since there is no 24 bit signed int data type
	struct Measurements res;
	res.x = 0;
	res.y = 0;
	// res.z=0;
	res.ret = 1;

	if (use_drdy)
	{
		if(timeout>0){
			int ret = wait_drdy(timeout);
			if (ret != 1)
			{
				res.ret = ret;
				return res;
			}
		}
	}
	else
		while (!get_measurement_ready())
			;
	uint8_t rx[7] = {
		0,
	};
	read_addr(0xA4, 6, rx);

	if ((rx[1] & 0x80) == 0x80)
	{
		res.x = -(int32_t)((~((uint32_t)rx[1] << 16 | (uint32_t)rx[2] << 8 | (uint32_t)rx[3]) + 1) & 0x007FFFFF);
	}
	else
		res.x = ((uint32_t)(rx[1] & 0x7F) << 16 | (uint32_t)(rx[2]) << 8 | (uint32_t)rx[3]);

	if ((rx[4] & 0x80) == 0x80)
	{
		res.y = -(int32_t)((~((uint32_t)rx[4] << 16 | (uint32_t)rx[5] << 8 | (uint32_t)rx[6]) + 1) & 0x007FFFFF);
	}
	else
		res.y = ((uint32_t)(rx[4] & 0x7F) << 16 | (uint32_t)(rx[5]) << 8 | (uint32_t)rx[6]);
	return res;
}

struct MeasurementsUt get_measurement_ut(uint16_t cycleCount)
{
	float gain = (0.3671 * (float)cycleCount) + 1.5; // linear equation to calculate the gain from cycle count
	struct Measurements res = get_measurement_poll();
	struct MeasurementsUt res_ut;
	res_ut.x = (float)res.x / gain;
	res_ut.y = (float)res.y / gain;
	// res_ut.z=(float)res.z/gain;
	return res_ut;
}

void set_continuous_measurement(bool enable)
{
	uint8_t tx[1];
	if (enable)
		tx[0] = 0x31;
	// tx[0]= 0x79;
	else
		tx[0] = 0x00;
	write_addr(RM3100_CMM_REG, 1, tx);
	return;
}

void set_tmrc(uint8_t tmrc_value)
{
	uint8_t tx[1] = {tmrc_value};
	write_addr(RM3100_TMRC_REG, 1, tx);
	return;
}
void poll_single_measurement()
{
	uint8_t tx[1] = {0x70};
	write_addr(RM3100_POLL_REG, 1, tx);
	return;
}

struct Measurements single_measurement()
{
	poll_single_measurement();
	while (!get_measurement_ready())
		;
	return get_measurement_poll();
}
struct MeasurementsUt single_measurement_ut(uint16_t cycleCount)
{
	poll_single_measurement();
	while (!get_measurement_ready())
		;
	return get_measurement_ut(cycleCount);
}
// newCC is the new cycle count value (16 bits) to change the data acquisition
void change_cycle_count(uint16_t newCC)
{
	uint8_t CCMSB = (newCC & 0xFF00) >> 8; // get the most significant byte
	uint8_t CCLSB = newCC & 0xFF;		   // get the least significant byte
	uint8_t tx[6] = {CCMSB, CCLSB, CCMSB, CCLSB, CCMSB, CCLSB};
	write_addr(RM3100_CCX1_REG, 6, tx);
}

int read_handshake()
{
	uint8_t rx[2] = {
		0,
	};
	read_addr(RM3100_HSHAKE_REG, 1, rx);
	return rx[1];
}

int start_mag()
{
	int ret = 0;
	int fd_local;
	fd_local = open(device, O_RDWR);
	if (fd_local < 0)
	{
		pabort("can't open device");
		return -1;
	}

	/*
	 * spi mode
	 */
	ret = ioctl(fd_local, SPI_IOC_WR_MODE, &mode);
	if (ret == -1)
	{
		pabort("can't set spi mode");
		return -1;
	}

	ret = ioctl(fd_local, SPI_IOC_RD_MODE, &mode);
	if (ret == -1)
	{
		pabort("can't get spi mode");
		return -1;
	}

	/*
	 * bits per word
	 */
	ret = ioctl(fd_local, SPI_IOC_WR_BITS_PER_WORD, &bits);
	if (ret == -1)
	{
		pabort("can't set bits per word");
		return -1;
	}

	ret = ioctl(fd_local, SPI_IOC_RD_BITS_PER_WORD, &bits);
	if (ret == -1)
	{
		pabort("can't get bits per word");
		return -1;
	}

	/*
	 * max speed hz
	 */
	ret = ioctl(fd_local, SPI_IOC_WR_MAX_SPEED_HZ, &speed);
	if (ret == -1)
	{
		pabort("can't set max speed hz");
		return -1;
	}

	ret = ioctl(fd_local, SPI_IOC_RD_MAX_SPEED_HZ, &speed);
	if (ret == -1)
	{
		pabort("can't get max speed hz");
		return -1;
	}
	fd = fd_local;
	// printf("spi mode: 0x%x\n", mode);
	// printf("bits per word: %d\n", bits);
	// printf("max speed: %d Hz (%d KHz)\n", speed, speed/1000);
	return 0;
}
int end_mag()
{

	close(fd);
	return 0;
}

int start_drdy()
{
	int ret;
	chip = gpiod_chip_open_by_name(chipname);
	if (!chip)
	{
		perror("Open chip failed\n");
		gpiod_chip_close(chip);
		return -1;
	}

	line = gpiod_chip_get_line(chip, line_num);
	if (!line)
	{
		perror("Get line failed\n");
		gpiod_line_release(line);
		gpiod_chip_close(chip);
		return -1;
	}

	ret = gpiod_line_request_rising_edge_events(line, CONSUMER);
	if (ret < 0)
	{
		perror("Rising edge request failed");
		return end_drdy();
	}
	return 0;
}
int end_drdy()
{
	gpiod_line_release(line);
	gpiod_chip_close(chip);
	return -1;
}
