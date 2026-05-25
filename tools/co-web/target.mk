# tools/co-web/target.mk -- Makefile hooks for the co-web playground.
# Included by the root Makefile via: -include $(TOOLS_MK)

EMSDK_DIR    ?= $(WORK_DIR)/extern/emsdk
WASM_BUILD_DIR = $(WORK_DIR)/build-wasm
CO_WEB_DIR   = $(WORK_DIR)/tools/co-web
CO_WEB_PORT  ?= 8080

.PHONY: co-web co-web-clean co-web-serve co-web-stop

co-web:
	@if [ ! -f $(WORK_DIR)/build/parser.tab.cc ]; then \
		echo "Error: native build required first (provides parser files)."; \
		echo "  Run: make build"; \
		exit 1; \
	fi
	@if [ ! -d $(EMSDK_DIR)/upstream ]; then \
		echo "Error: Emscripten SDK not found at $(EMSDK_DIR)."; \
		echo "  Install: git clone https://github.com/emscripten-core/emsdk.git $(EMSDK_DIR)"; \
		echo "           cd $(EMSDK_DIR) && ./emsdk install latest && ./emsdk activate latest"; \
		exit 1; \
	fi
	@echo "Building co-web (Choreo WebAssembly)..."
	@mkdir -p $(WASM_BUILD_DIR)
	@$(EMSDK_DIR)/upstream/emscripten/emcmake $(CMAKE) -S $(WORK_DIR) -B $(WASM_BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DENABLE_CUDA=OFF -DENABLE_CUTE=OFF -DENABLE_HIP=OFF \
		-DCHOREO_DEFAULT_TARGET=cc \
		-DCHOREO_BUILD_CO_MOCK=OFF \
		-DCHOREO_BUILD_CO_WEB=ON && \
	ninja -C $(WASM_BUILD_DIR) co-web && \
	cp $(WASM_BUILD_DIR)/co-web.js $(CO_WEB_DIR)/web/ && \
	cp $(WASM_BUILD_DIR)/co-web.wasm $(CO_WEB_DIR)/web/ && \
	echo "Build complete: tools/co-web/web/co-web.{js,wasm}"

co-web-clean:
	@rm -rf $(WASM_BUILD_DIR)
	@rm -f $(CO_WEB_DIR)/web/co-web.js $(CO_WEB_DIR)/web/co-web.wasm

co-web-serve: co-web
	@echo "Starting co-web playground on port $(CO_WEB_PORT)..."
	@cd $(CO_WEB_DIR)/web && python3 serve.py $(CO_WEB_PORT)

co-web-stop:
	@echo "Stopping co-web serve processes..."
	@ps aux | grep 'serve.py' | grep co-web | grep -v grep | awk '{print $$2}' | xargs -r kill 2>/dev/null || true
	@echo "Stopped."
