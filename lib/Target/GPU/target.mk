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
