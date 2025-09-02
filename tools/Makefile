TOOLCHAIN_DIR:=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))

#------------------------- GCU-2.x -------------------------#
GCU_REL_PATH=.
GCU_CMP_NAME=240524-gcu-compiler.tgz
GCU_CMP_PKG = $(TOOLCHAIN_DIR)/$(GCU_CMP_NAME)
GCU_CMP_PKG_MD5:=f6e0b029763acd1f66a192542a068ae5
CUR_GCU_CMP_PKG_MD5:=$(shell md5sum $(GCU_CMP_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu-sfc-kit:
	@if [ "$(CUR_GCU_CMP_PKG_MD5)" != "$(GCU_CMP_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu-sfc-kit; \
	else \
		echo "$(GCU_CMP_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu-sfc-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_REL_PATH)/$(GCU_CMP_NAME) -o $(GCU_CMP_PKG);\

install-gcu-sfc-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	cd $(TOOLCHAIN_DIR) && tar -zvxf $(GCU_CMP_PKG);

setup-gcu-sfc-kit: check-gcu-sfc-kit
	@if [ "$(CUR_GCU_CMP_PKG_MD5)" != "$(GCU_CMP_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu-sfc-kit; \
	fi

GCU_PLATFORM_NAME=TopsPlatform_1.0.1.6-a1e560_deb_amd64.run
GCU_PLATFORM_PKG=$(TOOLCHAIN_DIR)/$(GCU_PLATFORM_NAME)
GCU_PLATFORM_PKG_MD5:=216051f60566227b6b95bf7502a70178
CUR_GCU_PLATFORM_PKG_MD5:=$(shell md5sum $(GCU_PLATFORM_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu-platform-kit:
	@if [ "$(CUR_GCU_PLATFORM_PKG_MD5)" != "$(GCU_PLATFORM_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu-platform-kit; \
	else \
		echo "$(GCU_PLATFORM_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu-platform-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_REL_PATH)/$(GCU_PLATFORM_NAME) -o $(GCU_PLATFORM_PKG);\
	chmod +x $(GCU_PLATFORM_PKG);

install-gcu-platform-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	$(GCU_PLATFORM_PKG) -y -C topsruntime,topscc --install-dir $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/opt/tops/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/opt;

setup-gcu-platform-kit: check-gcu-platform-kit
	@if [ "$(CUR_GCU_PLATFORM_PKG_MD5)" != "$(GCU_PLATFORM_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu-platform-kit; \
	fi

GCU_FACTOR_NAME=topsfactor_3.0.1-1_amd64.deb
GCU_FACTOR_PKG = $(TOOLCHAIN_DIR)/$(GCU_FACTOR_NAME)
GCU_FACTOR_PKG_MD5:=03a257e7069cc4bb42270103e3616438
CUR_GCU_FACTOR_PKG_MD5:=$(shell md5sum $(GCU_FACTOR_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu-factor-kit:
	@if [ "$(CUR_GCU_FACTOR_PKG_MD5)" != "$(GCU_FACTOR_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu-factor-kit; \
	else \
		echo "$(GCU_FACTOR_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu-factor-kit:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_REL_PATH)/$(GCU_FACTOR_NAME) -o $(GCU_FACTOR_PKG);\

install-gcu-factor-kit:
	dpkg-deb -x $(GCU_FACTOR_PKG) $(TOOLCHAIN_DIR)/; \
	rsync -av $(TOOLCHAIN_DIR)/usr/* $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/local/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/usr/;
	rm -fr $(TOOLCHAIN_DIR)/local/;

setup-gcu-factor-kit: check-gcu-factor-kit
	@if [ "$(CUR_GCU_FACTOR_PKG_MD5)" != "$(GCU_FACTOR_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu-factor-kit; \
	fi

#------------------------- GCU-3.x -------------------------#
GCU3_REL_PATH=250508/gcu30
GCU3_PLATFORM_NAME=TopsPlatform_1.4.20250310-18636a_deb_amd64.run
GCU3_PLATFORM_PKG=$(TOOLCHAIN_DIR)/$(GCU3_PLATFORM_NAME)
GCU3_PLATFORM_PKG_MD5:=4220f099f1c50313ccd82823aaf34683
CUR_GCU3_PLATFORM_PKG_MD5:=$(shell md5sum $(GCU3_PLATFORM_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu3-platform-kit:
	@if [ "$(CUR_GCU3_PLATFORM_PKG_MD5)" != "$(GCU3_PLATFORM_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu3-platform-kit; \
	else \
		echo "$(GCU3_PLATFORM_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu3-platform-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU3_REL_PATH)/$(GCU3_PLATFORM_NAME) -o $(GCU3_PLATFORM_PKG);\
	chmod +x $(GCU3_PLATFORM_PKG);

install-gcu3-platform-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	$(GCU3_PLATFORM_PKG) -y -C topsruntime,topscc --install-dir $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/opt/tops/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/opt;

# specific used for package testing
install-gcu3-topscc: check-gcu3-platform-kit
	sudo $(GCU3_PLATFORM_PKG) -y -C topsruntime,topscc

setup-gcu3-platform-kit: check-gcu3-platform-kit
	@if [ "$(CUR_GCU3_PLATFORM_PKG_MD5)" != "$(GCU3_PLATFORM_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu3-platform-kit; \
	fi

GCU3_FACTOR_REL_PATH=241230/gcu30
GCU3_FACTOR_NAME=topsfactor_3.3.20241230-1_amd64.deb
GCU3_FACTOR_PKG = $(TOOLCHAIN_DIR)/$(GCU3_FACTOR_NAME)
GCU3_FACTOR_PKG_MD5:=a6e2e54a421a15a8b732fa7a24aa69b9
CUR_GCU3_FACTOR_PKG_MD5:=$(shell md5sum $(GCU3_FACTOR_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu3-factor-kit:
	@if [ "$(CUR_GCU3_FACTOR_PKG_MD5)" != "$(GCU3_FACTOR_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu3-factor-kit; \
	else \
		echo "$(GCU3_FACTOR_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu3-factor-kit:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU3_FACTOR_REL_PATH)/$(GCU3_FACTOR_NAME) -o $(GCU3_FACTOR_PKG);\

install-gcu3-factor-kit:
	dpkg-deb -x $(GCU3_FACTOR_PKG) $(TOOLCHAIN_DIR)/; \
	rsync -av $(TOOLCHAIN_DIR)/usr/* $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/local/* $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/opt/tops/* $(TOOLCHAIN_DIR); \
	ln -sf `pwd`/bin `pwd`/bin/clang11; \
	rm -fr $(TOOLCHAIN_DIR)/usr/; \
	rm -fr $(TOOLCHAIN_DIR)/local/; \
	rm -fr $(TOOLCHAIN_DIR)/opt;

setup-gcu3-factor-kit: check-gcu3-factor-kit
	@if [ "$(CUR_GCU3_FACTOR_PKG_MD5)" != "$(GCU3_FACTOR_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu3-factor-kit; \
	fi

install-gcu-sim:
	@if [ ! -f "$(TOOLCHAIN_DIR)/lib/libgcusim.so" ]; then \
		wget http://artifact.enflame.cn/artifactory/release_center/GCUSIM/1.0.79/libgcusim.so; \
		mv libgcusim.so $(TOOLCHAIN_DIR)/lib/libgcusim.so; \
	fi;

#-----------------------------------------------------------#
ACORE_PATH=9dca97c
ACORE_NAME=topsacore_3.3.20250822-9dca97c.tar.gz
ACORE_URL=http://artifact.enflame.cn/artifactory/module_package/acore/
ACORE_PKG=$(TOOLCHAIN_DIR)/$(ACORE_NAME)
ACORE_PKG_MD5=06b0bb09ae0670ba8caf99f146a9b00c
CUR_ACORE_PKG_MD5:=$(shell md5sum $(ACORE_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-acore:
	@if [ "$(ACORE_PKG_MD5)" != "$(CUR_ACORE_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the acore package..."; \
		$(MAKE) download-acore; \
	else \
		echo "$(ACORE_PKG) MD5 hash matches. No need to download."; \
	fi

download-acore:
	wget -P $(TOOLCHAIN_DIR) $(ACORE_URL)/$(ACORE_PATH)/$(ACORE_NAME)

install-acore:
	tar -zvxf $(TOOLCHAIN_DIR)/$(ACORE_NAME) $(TOOLCHAIN_DIR)/

setup-acore: check-acore
	@if [ "$(CUR_ACORE_PKG_MD5)" != "$(ACORE_PKG_MD5)"  ]; then \
	    $(MAKE) install-acore; \
	fi


gcu2-kmd: check-gcu-platform-kit
	sudo $(GCU_PLATFORM_PKG) -y -C enflame

# --force-asic is applied for some specific environment
gcu3-kmd: check-gcu3-platform-kit
	sudo $(GCU3_PLATFORM_PKG) -y -C enflame --force-asic

gcu2-kit: setup-gcu-sfc-kit setup-gcu-platform-kit setup-gcu-factor-kit
gcu3-kit: setup-gcu3-platform-kit setup-gcu3-factor-kit
gcu-sim: install-gcu-sim

gcu2-install: install-gcu-sfc-kit install-gcu-platform-kit install-gcu-factor-kit
gcu3-install: install-gcu3-platform-kit install-gcu3-factor-kit
