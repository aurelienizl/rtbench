
# Raspberry Pi 4 Kernel (rpi-6.12.y) — Dockerized Cross-Compile

Build a 64-bit Raspberry Pi 4 kernel (6.12.y) inside Docker, using an **out-of-tree** build, a small **config fragment** for RT tuning, and produce a ready-to-install tarball (kernel, DTBs, overlays, modules, final config).

---

## Contents

- `Dockerfile` — Ubuntu 24.04 image with cross toolchain, RPi kernel source (branch `rpi-6.12.y`), and `build.sh` as ENTRYPOINT.
- `build.sh` — Does the out-of-tree build, optional fragment merge, staging, and packaging.
- `fragment.config` — Sample config fragment (RT-oriented) baked into the image and used to seed `/work/fragment.config` if none is provided.

> Defaults are **hard-coded for Raspberry Pi 4 (arm64)**. You can override via env vars if needed (see **Advanced**).

---

## Prerequisites

- Docker installed on your host
- A working directory on your host (e.g., `rtpi/`)

---

## Quick Start (recommended)

```bash
# 0) In your project folder
docker build -t rpi4-kernel:6.12 .

# 1) Create host folders for build outputs
mkdir -p build out

# 2) (Optional) create/adjust your fragment locally
#    If omitted, the image’s baked template will be copied to /work/fragment.config.
nano fragment.config ./fragment.config  

# 3) Run the build (artifacts land in ./out and ./build)
docker run --rm -it \
  -v "$PWD/build:/work/build" \
  -v "$PWD/out:/work/out" \
  -v "$PWD/fragment.config:/work/fragment.config:ro" \
  rpi4-kernel:6.12
````

### Outputs

* `out/rpi4-<kernelrelease>.tar.gz` — **Install this on the Pi**
* `out/staging/` — Expanded tree (`boot/`, `lib/modules/`)
* `out/staging/boot/config-<kernelrelease>` — The final resolved `.config`

---

## Install on the Pi (manual, VFAT-safe)

Use the provided kernel_installer.sh file.

```bash
sudo bash kernel_installer.sh rpi4-<kernelrelease>.tar.gz
```

### Verify after reboot

```bash
uname -r
ls /lib/modules/$(uname -r)

# If you enabled IKCONFIG in your fragment:
[ -r /proc/config.gz ] && zcat /proc/config.gz | grep -E 'PREEMPT(_RT)?=|CONFIG_HZ_1000=' || echo "no /proc/config.gz"

# True PREEMPT_RT kernels expose this:
[ -e /sys/kernel/realtime ] && cat /sys/kernel/realtime || echo "no /sys/kernel/realtime"
```

---

## Customizing the build

### Config fragment (`fragment.config`)

This file is merged onto `bcm2711_defconfig` before `olddefconfig`. Example RT-leaning fragment:

```text
CONFIG_EXPERT=y
CONFIG_PREEMPT=y
CONFIG_PREEMPT_RT=y
CONFIG_PREEMPT_DYNAMIC=n
CONFIG_HIGH_RES_TIMERS=y
CONFIG_HZ_1000=y
CONFIG_IKCONFIG=y
CONFIG_IKCONFIG_PROC=y
# CONFIG_SCHED_AUTOGROUP is not set
# CONFIG_RT_GROUP_SCHED is not set
```

Mount your fragment into the container as a **file**:

```bash
-v "$PWD/fragment.config:/work/fragment.config:ro"
```

> ⚠️ Do **not** mount a **directory** at `/work/fragment.config` (or the merge will fail).

---

## Advanced (optional)

### Override defaults

You can override these via `-e` when running the container:

* `BRANCH` (default: `rpi-6.12.y`)
* `BUILD_DIR` (default: `/work/build`)
* `OUT_DIR` (default: `/work/out`)
* `ARCH` (default: `arm64`)
* `CROSS_COMPILE` (default: `aarch64-linux-gnu-`)
* `KERNEL_IMAGE` (default: `Image`)
* `OUT_KERNEL` (default: `kernel8.img`)
* `FRAG_FILE` (default: `/work/fragment.config`)

Example:

```bash
docker run --rm -it \
  -e BRANCH=rpi-6.12.y \
  -e OUT_KERNEL=mykernel8.img \
  -v "$PWD/build:/work/build" -v "$PWD/out:/work/out" \
  -v "$PWD/fragment.config:/work/fragment.config:ro" \
  rpi4-kernel:6.12
```

### Raw, manual debug inside a shell

```bash
docker run --rm -it --entrypoint bash rpi4-kernel:6.12

export ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
cd /src/linux
make O=/work/build bcm2711_defconfig
KCONFIG_CONFIG=/work/build/.config scripts/kconfig/merge_config.sh -m -O /work/build /work/build/.config /work/fragment.config
make O=/work/build olddefconfig
make -j"$(nproc)" O=/work/build Image modules dtbs
```

---

## Troubleshooting

* **Container “stops after defconfig/olddefconfig”**
  Likely not a Docker bug. If you were piping `yes "" | make olddefconfig` under `set -o pipefail`, `yes` can SIGPIPE (exit 141) → script stops. This project avoids that pipe and runs `olddefconfig` directly.

* **Permission denied / building in-tree**
  Always build **out-of-tree** (`O=/work/build`) so writes go to your mounted dir, and it works fine even when the container runs as your host UID.

* **merge\_config temp files error**
  Run merges from a **writable** dir (`/work/build`) and point `KCONFIG_CONFIG` at `/work/build/.config` (the script does this for you).

* **“cat: /work/.config: Is a directory”**
  You mounted a directory where a **file** was expected. Mount a file to `/work/fragment.config`.

* **VFAT `/boot/firmware` rsync errors**
  Use `--no-owner --no-group --no-perms` when copying to the boot partition.

* **System is using U-Boot/GRUB**
  If your `config.txt` had `kernel=u-boot.bin`, switching to `kernel=kernel8.img` moves to direct firmware boot. If you want to keep U-Boot/extlinux, use a packaging route for /boot entries instead.

* **BTF/pahole issues**
  We install `dwarves`. If you still hit BTF errors, disable `CONFIG_DEBUG_INFO_BTF` in your fragment or ensure pahole is recent.

---

## Cleaning up

```bash
# host
rm -rf build out
docker image rm rpi4-kernel:6.12
docker system prune
```

---

## License

This project assembles a build environment; the Linux kernel is licensed under GPLv2. Your `fragment.config` and scripts here are intended as examples.

```
::contentReference[oaicite:0]{index=0}