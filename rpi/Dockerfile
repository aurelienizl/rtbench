# Dockerfile — RPi4, branch rpi-6.12.y
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG RPI_KERNEL_BRANCH=rpi-6.12.y

RUN apt-get update && apt-get install -y \
    git make bc bison flex libssl-dev libncurses-dev \
    kmod cpio rsync dwarves device-tree-compiler \
    xz-utils lz4 zstd nano \
    crossbuild-essential-arm64 python3-minimal \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 --branch ${RPI_KERNEL_BRANCH} https://github.com/raspberrypi/linux.git /src/linux

ENV ARCH=arm64 \
    CROSS_COMPILE=aarch64-linux-gnu- \
    KERNEL_DEFCONFIG=bcm2711_defconfig \
    KERNEL_IMAGE=Image \
    OUT_KERNEL=kernel8.img \
    WORK_MOUNT=/work

# Entry script + baked-in fragment template
COPY build.sh /build.sh
RUN chmod +x /build.sh
COPY fragment.config /fragment.config

ENTRYPOINT [ "/build.sh" ]
