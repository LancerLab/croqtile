CFLAGS += -D__CHOREO_CUDA_DIR__="$(TOOLCHAIN_DIR)"
CFLAGS += -D__CHOREO_CUTE_DIR__="$(TOOLCHAIN_DIR)/cutlass"

$(LGY_BUILD_DIR)/choreo_cute_header.inc : $(RT_DIR)/choreo_cute.h
	echo "#ifndef __CHOREO_CUTE_RUNTIME_HEADER_H__" > $@
	echo "#define __CHOREO_CUTE_RUNTIME_HEADER_H__" >> $@
	echo -n "static const char* __choreo_cute_header_as_string = R\"(" >> $@
	cat $< >> $@
	echo ")\";" >> $@
	echo "#endif // __CHOREO_CUTE_RUNTIME_HEADER_H__" >> $@

HEADER_FILES += $(LGY_BUILD_DIR)/choreo_cute_header.inc

# =============================================================================
# OSS Compliance Scan (works on both main and oss/main)
# =============================================================================
OSS_SCAN_DIR := $(SCRIPT_DIR)/oss
OSS_SCAN_SH  := $(OSS_SCAN_DIR)/oss-scan.sh

# Only define scan targets if the script exists on this branch
ifneq ($(wildcard $(OSS_SCAN_SH)),)

.PHONY: oss-scan oss-scan-staged oss-scan-diff

oss-scan:
	@bash $(OSS_SCAN_SH) --tree HEAD

oss-scan-staged:
	@bash $(OSS_SCAN_SH) --staged

oss-scan-diff:
	@if [ -z "$(COMMIT)" ]; then echo "Usage: make oss-scan-diff COMMIT=<sha>"; exit 1; fi
	@bash $(OSS_SCAN_SH) --diff $(COMMIT)

endif
