FTP_SERVER ?= 172.16.11.18
GITLAB_SERVER ?= git.enflame.cn

SETUP_TARGET_DEPENDS += setup-clang-format
SETUP_TARGET_DEPENDS += setup-skills
#SETUP_TARGET_DEPENDS += setup-gcu-acore
CHOREO_DEFAULT_TARGET = topscc
CLANG_FORMAT:=$(WORK_DIR)/extern/clang-format-19-1-2

FILECHECK:=$(TOOLCHAIN_DIR)/bin/FileCheck
PACKAGE_NAME=choreo_toolchain_250930.tgz
SUPPORT_PKG =$(TOOLCHAIN_DIR)/$(PACKAGE_NAME)
PACKAGE_MD5:=a4297fca634dcdda3d08c467550d4b22
CUR_PKG_MD5:=$(shell md5sum $(SUPPORT_PKG) 2>/dev/null| cut -d ' ' -f 1)
BISON_ENV:=BISON_PKGDATADIR=$(TOOLCHAIN_DIR)/shared/bison/
BISON:=$(BISON_ENV) $(BISON_BIN)
CFLAGS += -D__CHOREO_TOPSCC_DIR__="$(TOOLCHAIN_DIR)"

.PHONY: setup-gcu2 setup-gcu3 setup-gcu4sim setup-gcu5sim

check-choreo-kit:
	@if [ "$(CUR_PKG_MD5)" != "$(PACKAGE_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the supporting package..."; \
		$(MAKE) download-choreo-kit; \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi;

download-choreo-kit:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(PACKAGE_NAME) -o $(SUPPORT_PKG);\

install-choreo-kit: check-choreo-kit
	cd $(TOOLCHAIN_DIR) && tar -zvxf $(SUPPORT_PKG); \
	chmod +x $(BISON_BIN); \
	ln -sf $(WORK_DIR)/extern/bin/not.sh tests

setup-choreo-kit: check-choreo-kit
	@if [ "$(CUR_PKG_MD5)" != "$(PACKAGE_MD5)"  ]; then \
	  $(MAKE) install-choreo-kit; \
	fi;

setup-clang-format: check-clang-format
	chmod +x $(CLANG_FORMAT)

# setup-skills: internal-only. Clones/updates croqtile-skills, then
# symlinks .claude, .codex, .github into the choreo workspace.
setup-skills:
	@if [ ! -d $(TOOLCHAIN_DIR)/croqtile-skills ]; then \
	  echo "[setup-skills] cloning croqtile-skills..."; \
	  git clone git@$(GITLAB_SERVER):era-dev/croqtile-skills.git $(TOOLCHAIN_DIR)/croqtile-skills; \
	fi
	@$(MAKE) -C $(TOOLCHAIN_DIR)/croqtile-skills update
	@$(MAKE) -C $(TOOLCHAIN_DIR)/croqtile-skills setup-skills

# -----------------------------------------------------------------------
# cmp-perf  --  CI benchmark targets for choreo+topscc vs coir+topscc
#
# These targets compare the two GCU compilation paths:
#   1. choreo -t topscc  (AST pipeline)
#   2. cocc -t topscc    (CoIR MLIR pipeline)
#
# Targets:
#   make cmp-perf          Run compile + execute + stats (fail-fast)
#   make cmp-perf-compile  Compile all benchmarks (emit-source only)
#   make cmp-perf-execute  Compile and execute all benchmarks on GCU
#   make cmp-perf-stats    Collect and compare --stats from both paths
#
# Variables:
#   CMP_PERF_FILTER   Limit to files matching a pattern (e.g. "matmul")
#   CMP_PERF_JOBS      Parallelism for compilation (default: 1)
# -----------------------------------------------------------------------
CMP_PERF_FILTER ?=
CMP_PERF_JOBS   ?= 1
CMP_PERF_SCRIPT := $(SCRIPT_DIR)/cmp-perf-ci.sh
CMP_PERF_STATS  := $(SCRIPT_DIR)/cmp-perf-stats.sh

.PHONY: cmp-perf cmp-perf-compile cmp-perf-execute cmp-perf-stats

cmp-perf: coir
	@echo "[cmp-perf] Phase 1/3: compile check..."
	@bash $(CMP_PERF_SCRIPT) compile --jobs $(CMP_PERF_JOBS) \
	  $(if $(CMP_PERF_FILTER),--filter "$(CMP_PERF_FILTER)",)
	@echo "[cmp-perf] Phase 2/3: execute check..."
	@bash $(CMP_PERF_SCRIPT) execute --jobs $(CMP_PERF_JOBS) \
	  $(if $(CMP_PERF_FILTER),--filter "$(CMP_PERF_FILTER)",)
	@echo "[cmp-perf] Phase 3/3: stats..."
	@bash $(CMP_PERF_STATS) stats-collect \
	  $(if $(CMP_PERF_FILTER),--filter "$(CMP_PERF_FILTER)",) \
	  --output $(BUILD_DIR)/cmp-perf-stats.jsonl
	@cat $(BUILD_DIR)/cmp-perf-stats.jsonl | bash $(CMP_PERF_STATS) stats-compare
	@echo "[cmp-perf] All phases PASSED"

cmp-perf-compile: coir
	@bash $(CMP_PERF_SCRIPT) compile --jobs $(CMP_PERF_JOBS) \
	  $(if $(CMP_PERF_FILTER),--filter "$(CMP_PERF_FILTER)",)

cmp-perf-execute: coir
	@bash $(CMP_PERF_SCRIPT) execute --jobs $(CMP_PERF_JOBS) \
	  $(if $(CMP_PERF_FILTER),--filter "$(CMP_PERF_FILTER)",)

cmp-perf-stats: coir
	@bash $(CMP_PERF_STATS) stats-collect \
	  $(if $(CMP_PERF_FILTER),--filter "$(CMP_PERF_FILTER)",) \
	  --output $(BUILD_DIR)/cmp-perf-stats.jsonl
	@cat $(BUILD_DIR)/cmp-perf-stats.jsonl | bash $(CMP_PERF_STATS) stats-compare

setup-gcu-acore:
	cd $(TOOLCHAIN_DIR) && $(MAKE) setup-acore FTP_SERVER=$(FTP_SERVER)

setup-cuda:
	cd $(TOOLCHAIN_DIR) && $(MAKE) setup-cuda FTP_SERVER=$(FTP_SERVER)

setup-gcu2: setup-core
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu3: setup-core setup-gcu-acore
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu4: setup-core
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu4-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu5: setup-core
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu5-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu4sim: setup-core
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu4-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu5sim: setup-core
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu5-kit FTP_SERVER=$(FTP_SERVER)

resetup-gcu2: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-install FTP_SERVER=$(FTP_SERVER)

resetup-gcu3: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-install FTP_SERVER=$(FTP_SERVER)

resetup-gcu4: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu4-install FTP_SERVER=$(FTP_SERVER)

resetup-gcu5: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu5-install FTP_SERVER=$(FTP_SERVER)

gcu2-kmd:
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-kmd FTP_SERVER=$(FTP_SERVER)

gcu3-kmd:
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-kmd FTP_SERVER=$(FTP_SERVER)


CFORMAT_MD5=6ee59eba63782b362bc9ba1138911f3a
CFORMAT_NAME=clang-format-19-1-2
CUR_CFORMAT_MD5:=$(shell md5sum $(CLANG_FORMAT) 2>/dev/null| cut -d ' ' -f 1)

download-clang-format:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(CFORMAT_NAME) -o $(CLANG_FORMAT);\

check-clang-format:
	@if [ "$(CUR_CFORMAT_MD5)" != "$(CFORMAT_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the clang-format..."; \
		$(MAKE) download-clang-format; \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi;

# utils to serve Choreo Documents
MKDOCS_CMD = mkdocs serve --dev-addr=0.0.0.0:8000

serve-doc: stop-doc start-doc

stop-doc:
	@echo "Stopping existing mkdocs serve processes..."
	@ps aux | grep 'mkdocs serve' | grep -v grep | awk '{print $$2}' | xargs -r kill 2>/dev/null || true
	@echo "Old mkdocs serve processes stopped."

start-doc: docs
	@command -v mkdocs >/dev/null 2>&1 || { echo "ERROR: mkdocs not found. Install with: pip install --user mkdocs mkdocs-material"; exit 1; }
	@mkdocs --version >/dev/null 2>&1 || { echo "ERROR: mkdocs is installed but broken (check Python environment)."; echo "  Try: pip install --user --force-reinstall mkdocs mkdocs-material"; exit 1; }
	@echo "Starting mkdocs serve in the background..."
	@nohup $(MKDOCS_CMD) > /tmp/mkdocs-serve.log 2>&1 & \
	sleep 1; \
	if ps -p $$! >/dev/null 2>&1; then \
		echo "mkdocs serve started (pid $$!, log: /tmp/mkdocs-serve.log)"; \
		echo "  -> http://0.0.0.0:8000"; \
	else \
		echo "ERROR: mkdocs serve failed to start. Check /tmp/mkdocs-serve.log:"; \
		tail -5 /tmp/mkdocs-serve.log; \
		exit 1; \
	fi

status-doc:
	@echo "Checking mkdocs serve process..."
	@ps aux | grep 'mkdocs serve' | grep -v grep || echo "No mkdocs serve process is running."

# utils to publish packages to releases or package registry
publish-package: package
	@bash scripts/publish-package.sh

publish-release: package
	@bash scripts/publish-release.sh

publish-to-apex: package
	@bash scripts/publish-choreo-for-apex.sh

publish-to-topsop: package
	@bash scripts/publish-choreo-for-topsop.sh

publish-sdk: sdk-package
	pkg_name=$$(find $(REL_BUILD_DIR)/package/_CPack_Packages/Linux/DEB/ -name 'choreo-dev*.deb'); \
	sdk_name=$$(basename $$pkg_name); \
	md5sum $$pkg_name; \
	curl -T $$pkg_name ftp://$(FTP_SERVER)/\%2fdev/choreo-sdk/$$sdk_name --user ftp_era:Enflame@321

test-libra: release
	$(LIT) --sim=only tests/gcu/end2end && $(MAKE) standalone-test-with-cmake

# =============================================================================
# Sample Tests for topscc/elementwise
# =============================================================================

ELEMENTWISE_DIR = samples/topscc/elementwise
OPERATOR_NAMES = $(notdir $(basename $(wildcard $(ELEMENTWISE_DIR)/*.co)))
CHOREO_FLAGS = -gs -t topscc

sample-test: $(OPERATOR_NAMES:%=sample-test-%)

sample-test-%: $(ELEMENTWISE_DIR)/%.co
	@TMPDIR=$$(mktemp -d) && \
	echo -n "Testing $*... " && \
	if choreo $(CHOREO_FLAGS) $< -o $$TMPDIR/test.result > /dev/null 2>&1 && \
	   bash $$TMPDIR/test.result --execute > /dev/null 2>&1; then \
		echo "PASSED"; \
		ret=0; \
	else \
		echo "FAILED"; \
		ret=1; \
	fi; \
	rm -rf $$TMPDIR; \
	exit $$ret

sample-test-operator:
	@if [ -z "$(OPERATOR)" ]; then \
		echo "Usage: make sample-test-operator OPERATOR=operator_name"; \
		echo "Available operators: $(OPERATOR_NAMES)"; \
		exit 1; \
	fi
	@TMPDIR=$$(mktemp -d) && \
	echo -n "Testing $(OPERATOR)... " && \
	if choreo $(CHOREO_FLAGS) $(ELEMENTWISE_DIR)/$(OPERATOR).co -o $$TMPDIR/test.result > /dev/null 2>&1 && \
	   bash $$TMPDIR/test.result --execute > /dev/null 2>&1; then \
		echo "PASSED"; \
		ret=0; \
	else \
		echo "FAILED"; \
		ret=1; \
	fi; \
	rm -rf $$TMPDIR; \
	exit $$ret

run-samples: $(OPERATOR_NAMES:%=test-%)

# =============================================================================
# Open-Source Sync (oss/main <-> main)
# =============================================================================
# Internal workflow for syncing code to the public croqtile repository.
# Full guide: Documents/internal/oss-sync-developer-guide.md

OSS_SCRIPTS := $(SCRIPT_DIR)/oss
OSS_PUSH    := bash $(OSS_SCRIPTS)/oss-push.sh
OSS_PULL    := bash $(OSS_SCRIPTS)/oss-pull.sh
OSS_SCAN    := bash $(OSS_SCRIPTS)/oss-scan.sh
OSS_PSCAN   := bash $(OSS_SCRIPTS)/oss-pull-scan.sh
OSS_WATCH   := bash $(OSS_SCRIPTS)/oss-watch.sh
OSS_SETUP   := bash $(OSS_SCRIPTS)/oss-setup.sh
COMMIT     ?=
RANGE      ?=

SYNC_ALL    := bash $(OSS_SCRIPTS)/sync_all.sh

# oss-scan targets are defined in lib/Target/GPU/target.mk (shared by both branches)
.PHONY: oss-push oss-push-last \
        oss-push-range oss-push-dry oss-catchup oss-catchup-dry \
        oss-pull oss-pull-last oss-pull-catchup oss-pull-catchup-dry \
        oss-pull-scan oss-watch \
        oss-setup oss-status oss-help oss-sync-check oss-sync-check-verbose \
        sync-all sync-all-once

oss-push:
	@if [ -z "$(COMMIT)" ]; then \
		echo "Usage:"; \
		echo "  make oss-push COMMIT=<sha>          # single commit"; \
		echo "  make oss-push-last                   # last commit on main"; \
		echo "  make oss-push-range RANGE=a..b       # range of commits"; \
		echo ""; \
		echo "See: Documents/internal/oss-sync-developer-guide.md"; \
		exit 1; \
	fi
	@$(OSS_PUSH) $(COMMIT)

oss-push-last:
	@echo "Pushing last commit on main to oss/main..."
	@$(OSS_PUSH) --interactive HEAD

oss-push-range:
	@if [ -z "$(RANGE)" ]; then echo "Usage: make oss-push-range RANGE=<from>..<to>"; exit 1; fi
	@$(OSS_PUSH) --range $(RANGE)

oss-push-dry:
	@if [ -z "$(COMMIT)" ] && [ -z "$(RANGE)" ]; then \
		echo "Usage:"; \
		echo "  make oss-push-dry COMMIT=<sha>"; \
		echo "  make oss-push-dry RANGE=a..b"; \
		exit 1; \
	fi
	@if [ -n "$(RANGE)" ]; then \
		$(OSS_PUSH) -n --range $(RANGE); \
	else \
		$(OSS_PUSH) -n $(COMMIT); \
	fi

oss-catchup:
	@echo "Catching up oss/main with all unsynced main commits..."
	@$(OSS_PUSH) --catchup

oss-catchup-dry:
	@echo "Preview: catching up oss/main with all unsynced main commits..."
	@$(OSS_PUSH) -n --catchup

oss-pull:
	@if [ -z "$(COMMIT)" ]; then \
		echo "Usage:"; \
		echo "  make oss-pull COMMIT=<sha>     # pull specific commit"; \
		echo "  make oss-pull-last             # pull last unpulled commit"; \
		echo ""; \
		echo "See: Documents/internal/oss-sync-developer-guide.md"; \
		exit 1; \
	fi
	@$(OSS_PULL) $(COMMIT)

oss-pull-last:
	@echo "Pulling last unpulled commit from oss/main..."
	@$(OSS_PULL) --last

oss-pull-catchup:
	@echo "Catching up main with all unpulled oss/main commits..."
	@$(OSS_PULL) --catchup

oss-pull-catchup-dry:
	@echo "Preview: catching up main with all unpulled oss/main commits..."
	@$(OSS_PULL) -n --catchup

oss-pull-scan:
	@$(OSS_PSCAN) --fetch

oss-watch:
	@$(OSS_WATCH)

oss-setup:
	@$(OSS_SETUP)

oss-status:
	@echo "=== OSS Branch Status ==="
	@echo "oss/main tip:"; git log oss/main --oneline -3 2>/dev/null || echo "  (branch not found -- run: make oss-setup)"
	@echo ""; echo "main tip:"; git log main --oneline -3
	@echo ""; echo "Remotes:"
	@git remote -v | grep -E '(oss-shadow|public)' || echo "  (no oss remote -- run: make oss-setup)"

oss-sync-check:
	@bash $(OSS_SCRIPTS)/oss-sync-check.sh

oss-sync-check-verbose:
	@bash $(OSS_SCRIPTS)/oss-sync-check.sh -v

oss-help:
	@echo "Open-Source Sync Commands"
	@echo "========================="
	@echo ""
	@echo "  make oss-scan                   Scan oss/main branch for violations"
	@echo "  make oss-scan-staged            Scan staged changes for violations"
	@echo "  make oss-scan-diff COMMIT=sha   Scan a single commit"
	@echo ""
	@echo "  make oss-push COMMIT=sha        Push one commit to oss/main"
	@echo "  make oss-push-last              Push HEAD to oss/main"
	@echo "  make oss-push-range RANGE=a..b  Push a range of commits"
	@echo "  make oss-push-dry COMMIT=sha    Dry-run (preview only)"
	@echo "  make oss-push-dry RANGE=a..b    Dry-run a range"
	@echo "  make oss-catchup                Auto-sync all new main commits to oss/main"
	@echo "  make oss-catchup-dry            Preview what oss-catchup would do"
	@echo ""
	@echo "  make oss-pull COMMIT=sha        Pull from oss/main to main"
	@echo "  make oss-pull-last              Pull last unpulled oss/main commit"
	@echo "  make oss-pull-catchup           Auto-pull all unpulled oss/main commits to main"
	@echo "  make oss-pull-catchup-dry       Preview what oss-pull-catchup would do"
	@echo "  make oss-pull-scan              Fetch public + scan for conflicts"
	@echo "  make oss-watch                  Start periodic sync watcher (WSL)"
	@echo ""
	@echo "  make oss-setup                  Initialize oss remote + branch"
	@echo "  make oss-status                 Show sync status"
	@echo "  make oss-sync-check             Verify file consistency main vs oss/main"
	@echo ""
	@echo "  make sync-all                   Start unified sync daemon (2-min loop)"
	@echo "  make sync-all-once              Run one unified sync cycle"
	@echo "  make force-sync BRANCH=oss/main Force-sync branch to all remotes"
	@echo ""
	@echo "Full guide: Documents/internal/oss-sync-developer-guide.md"

# =============================================================================
# Unified Sync Daemon (origin <-> mirror + main <-> oss/main <-> GitHub)
# =============================================================================

FORCE_SYNC := bash $(SCRIPT_DIR)/force_sync_branch.sh

.PHONY: sync-all sync-all-once force-sync force-sync-dry

sync-all:
	@echo "Starting unified sync daemon (origin <-> mirror + oss)..."
	@echo "Press Ctrl-C to stop."
	@$(SYNC_ALL) --log

sync-all-once:
	@$(SYNC_ALL) --once

force-sync:
	@if [ -z "$(BRANCH)" ]; then \
		echo "Usage: make force-sync BRANCH=<branch> [REMOTE=<remote>]"; \
		echo "       make force-sync BRANCH=oss/main              (all remotes)"; \
		echo "       make force-sync BRANCH=oss/main REMOTE=origin"; \
		exit 1; \
	fi
	@if [ -z "$(REMOTE)" ]; then \
		$(FORCE_SYNC) --all-remotes $(BRANCH); \
	else \
		$(FORCE_SYNC) $(BRANCH) $(REMOTE); \
	fi

force-sync-dry:
	@if [ -z "$(BRANCH)" ]; then \
		echo "Usage: make force-sync-dry BRANCH=<branch>"; \
		exit 1; \
	fi
	@if [ -z "$(REMOTE)" ]; then \
		$(FORCE_SYNC) -n --all-remotes $(BRANCH); \
	else \
		$(FORCE_SYNC) -n $(BRANCH) $(REMOTE); \
	fi
