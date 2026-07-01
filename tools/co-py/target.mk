# tools/co-py/target.mk -- Makefile hooks for the co-py Python bindings.
# Included by the root Makefile via: -include $(TOOLS_MK)

CO_PY_DIR    = $(WORK_DIR)/tools/co-py
CO_PY_PYTHON ?= python3

.PHONY: co-py co-py-clean co-py-test

co-py: build
	@echo "=== Building co-py (Python bindings) ==="
	$(CMAKE) -S $(WORK_DIR) -B $(BUILD_DIR) \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		-DCROQ_PROJECT="choreo;co-py"
	ninja -C $(BUILD_DIR) _core
	@echo "Done. Use: PYTHONPATH=$(CO_PY_DIR)/src $(CO_PY_PYTHON) -c 'import croq'"

co-py-test: co-py
	@echo "=== Running co-py tests ==="
	PYTHONPATH=$(CO_PY_DIR)/src \
	CHOREO_BIN=$(WORK_DIR)/build/choreo \
	$(CO_PY_PYTHON) -m pytest $(CO_PY_DIR)/tests/ -v

co-py-clean:
	@rm -f $(CO_PY_DIR)/src/croqtile/_core*.so
