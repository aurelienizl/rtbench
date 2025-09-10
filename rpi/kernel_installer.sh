#!/bin/bash
set -e

# Usage: ./update-kernel.sh /path/to/rpi4-6.12.45-v8+.tar.gz
if [ $# -lt 1 ]; then
  echo "Usage: $0 <kernel-tarball>"
  exit 1
fi

TARBALL="$1"

if [ ! -f "$TARBALL" ]; then
  echo "Error: tarball '$TARBALL' not found"
  exit 1
fi

BOOT=/boot
mountpoint -q /boot/firmware && BOOT=/boot/firmware
echo "Boot mount: $BOOT"

# Backup current config.txt (just in case)
sudo cp -a "$BOOT/config.txt" \
  "$BOOT/config.txt.bak.$(date +%Y%m%d-%H%M%S)" \
  2>/dev/null || true

# Extract to a temp dir
TMP="$(mktemp -d)"
trap 'sudo rm -rf "$TMP"' EXIT
tar xzf "$TARBALL" -C "$TMP"

# Copy boot files WITHOUT preserving owner/perms (VFAT friendly)
sudo rsync -rltDv --no-owner --no-group --no-perms "$TMP/boot/" "$BOOT/"

# Copy modules to rootfs (ext4)
if [ -d "$TMP/lib/modules" ]; then
  sudo rsync -a "$TMP/lib/modules/" /lib/modules/
fi

# Ensure firmware boots the new 64-bit kernel file
CFG="$BOOT/config.txt"
sudo sed -i -E '/^arm_64bit=|^kernel=/d' "$CFG"
printf '\n# Custom kernel from %s\narm_64bit=1\nkernel=kernel8.img\n' \
  "$(basename "$TARBALL")" | sudo tee -a "$CFG" >/dev/null

echo "NOTE: If this file previously contained 'kernel=u-boot.bin' (Ubuntu style)"
echo "You've now switched to direct firmware boot. That's fine; just be aware."
