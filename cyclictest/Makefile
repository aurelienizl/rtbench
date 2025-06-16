# Simple Makefile for cyclictest only

# --- User‐configurable variables ---
VERSION      ?= 2.8
CROSS_COMPILE?=
CC           := $(CROSS_COMPILE)gcc
CFLAGS       ?= -Wall -Werror -Wextra -Wno-sign-compare -Wno-unused-parameter \
                -Wno-error=format-truncation= -O2 -g
CPPFLAGS     := -D_GNU_SOURCE -Iinclude
LDFLAGS      ?=
LIBS         := -lrt -lpthread
# for NUMA support:
EXTRA_LIBS   ?= -lnuma

# --- Sources & targets ---
SRC          := cyclictest.c rt-error.c rt-numa.c rt-utils.c rt-sched.c histogram.c 
OBJ          := $(SRC:.c=.o)
TARGET       := cyclictest

# --- Rules ---
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(EXTRA_LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

install: $(TARGET)
	install -D -m 755 $(TARGET) /usr/local/bin/$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
