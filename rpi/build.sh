#!/usr/bin/env bash
set -euxo pipefail
trap 'echo "FAILED: ${BASH_COMMAND}" >&2' ERR

BRANCH="${BRANCH:-rpi-6.12.y}"
SRC="${SRC:-/src/linux}"
BUILD="${BUILD_DIR:-/work/build}"
OUT="${OUT_DIR:-/work/out}"
JOBS="${JOBS:-$(nproc)}"

export ARCH="${ARCH:-arm64}"
export CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"

mkdir -p "$BUILD" "$OUT" /work

# ---- fragment handling (file, not directory!) ----
FRAG_FILE="${FRAG_FILE:-/work/fragment.config}"   # preferred user path
BAKED_FRAG="/fragment.config"                     # template baked into the image

# guard: if user mounted a directory at FRAG_FILE path, fail clearly
if [[ -e "$FRAG_FILE" && -d "$FRAG_FILE" ]]; then
  echo "ERROR: FRAG_FILE ($FRAG_FILE) is a directory. Mount a FILE, e.g.:"
  echo "  -v \"\$PWD/fragment.config:/work/fragment.config:ro\""
  exit 2
fi

# seed from baked-in template if user file missing/empty
if [[ ! -s "$FRAG_FILE" && -s "$BAKED_FRAG" ]]; then
  cp -v "$BAKED_FRAG" "$FRAG_FILE"
fi

HAS_FRAG=0
[[ -s "$FRAG_FILE" ]] && HAS_FRAG=1

echo "==> SRC=$SRC BRANCH=$BRANCH"
echo "==> BUILD=$BUILD OUT=$OUT"
echo "==> FRAG_FILE=$FRAG_FILE (present=$HAS_FRAG)"
echo "==> ARCH=$ARCH CROSS_COMPILE=$CROSS_COMPILE JOBS=$JOBS"

# clone if missing (Dockerfile already cloned)
if [[ ! -d "$SRC/.git" ]]; then
  git clone --depth=1 --branch "$BRANCH" https://github.com/raspberrypi/linux.git "$SRC"
fi

# clean out-of-tree dir
make -C "$SRC" O="$BUILD" mrproper

# base config
make -C "$SRC" O="$BUILD" bcm2711_defconfig

# merge fragment (base is $BUILD/.config, fragment is $FRAG_FILE)
if (( HAS_FRAG )); then
  ( cd "$BUILD"
    KCONFIG_CONFIG="$BUILD/.config" \
    "$SRC/scripts/kconfig/merge_config.sh" -m -O "$BUILD" "$BUILD/.config" "$FRAG_FILE"
  )
else
  echo "==> No fragment provided; skipping merge"
fi

# finalize defaults (no yes| to avoid SIGPIPE with pipefail)
make -C "$SRC" O="$BUILD" olddefconfig

# build
make -j"$JOBS" -C "$SRC" O="$BUILD" "${KERNEL_IMAGE:-Image}" modules dtbs

# stage + package
STAGE="$OUT/staging"; mkdir -p "$STAGE/boot/overlays"

cp -v "$BUILD/arch/arm64/boot/${KERNEL_IMAGE:-Image}" "$STAGE/boot/${OUT_KERNEL:-kernel8.img}"
cp -v "$BUILD/arch/arm64/boot/dts/broadcom/"*.dtb "$STAGE/boot/" 2>/dev/null || true
cp -v "$BUILD/arch/arm64/boot/dts/overlays/"*.dtb* "$STAGE/boot/overlays/" 2>/dev/null || true
cp -v "$BUILD/arch/arm/boot/dts/overlays/"*.dtb*   "$STAGE/boot/overlays/" 2>/dev/null || true
cp -v "$BUILD/arch/arm64/boot/dts/overlays/README" "$STAGE/boot/overlays/" 2>/dev/null || true
cp -v "$BUILD/arch/arm/boot/dts/overlays/README"   "$STAGE/boot/overlays/" 2>/dev/null || true

make -C "$SRC" O="$BUILD" modules_install INSTALL_MOD_PATH="$STAGE"

KREL="$(make -s -C "$SRC" O="$BUILD" kernelrelease)"
cp -v "$BUILD/.config" "$STAGE/boot/config-$KREL"
( cd "$STAGE" && tar czf "$OUT/rpi4-${KREL}.tar.gz" . )

echo "Tarball: $OUT/rpi4-${KREL}.tar.gz"
