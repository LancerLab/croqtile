# tools/coir/target.mk -- Makefile targets for CoIR IR tooling
#
# Included automatically by the root Makefile via TOOLS_MK wildcard.

COIR_DEP_CONF = $(WORK_DIR)/tools/coir/llvm-dep.conf
COIR_LLVM_DIR = $(WORK_DIR)/extern/llvm-project
COIR_LLVM_SHASH := $(shell grep '^LLVM_SHASH=' $(COIR_DEP_CONF) | cut -d= -f2-)
COIR_LLVM_TAR := $(shell grep '^LLVM_TAR=' $(COIR_DEP_CONF) | cut -d= -f2-)
COIR_LLVM_URL := $(shell grep '^LLVM_URL=' $(COIR_DEP_CONF) | cut -d= -f2-)
COIR_LLVM_MD5 := $(shell grep '^LLVM_MD5=' $(COIR_DEP_CONF) | cut -d= -f2-)

# Download and extract LLVM/MLIR into extern/llvm-project/
.PHONY: setup-coir-deps
setup-coir-deps:
	@echo "=== Setting up LLVM/MLIR for CoIR tooling ==="
	@mkdir -p $(COIR_LLVM_DIR)
	@if [ ! -f "$(COIR_LLVM_DIR)/lib/cmake/mlir/MLIRConfig.cmake" ]; then \
		echo "Downloading LLVM/MLIR ($(COIR_LLVM_SHASH))..."; \
		cd $(WORK_DIR)/extern && \
		wget -q --show-progress -O $(COIR_LLVM_TAR) $(COIR_LLVM_URL) && \
		echo "Extracting into extern/llvm-project/..." && \
		tar -xzf $(COIR_LLVM_TAR) --strip-components=1 -C llvm-project && \
		rm -f $(COIR_LLVM_TAR) && \
		echo "LLVM/MLIR ready at $(COIR_LLVM_DIR)"; \
	else \
		echo "LLVM/MLIR already present at $(COIR_LLVM_DIR)"; \
	fi

# Build all CoIR tools (co2ir, coir-opt, coir-codegen, cocc)
.PHONY: coir
coir: build
	@echo "=== Building CoIR tools ==="
	cmake -S $(WORK_DIR) -B $(BUILD_DIR) \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCHOREO_BUILD_COIR=ON \
		-DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET)
	ninja -C $(BUILD_DIR) co2ir coir-opt coir-codegen cocc
	@echo "CoIR tools built:"
	@echo "  $(BUILD_DIR)/tools/coir/co2ir"
	@echo "  $(BUILD_DIR)/tools/coir/coir-opt"
	@echo "  $(BUILD_DIR)/tools/coir/coir-codegen"
	@echo "  $(BUILD_DIR)/tools/coir/cocc"

# Run CoIR lit tests
.PHONY: coir-test
coir-test: coir
	@echo "=== Running CoIR tests ==="
	$(LIT) tools/coir/tests/

# Clean CoIR build artifacts
.PHONY: coir-clean
coir-clean:
	@echo "=== Cleaning CoIR build artifacts ==="
	@if [ -d "$(BUILD_DIR)/tools/coir" ]; then \
		rm -rf $(BUILD_DIR)/tools/coir; \
		echo "Removed $(BUILD_DIR)/tools/coir"; \
	fi
