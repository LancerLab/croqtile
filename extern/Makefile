TOOLCHAIN_DIR:=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
GCU_SIM_DIR:=$(TOOLCHAIN_DIR)/gcu-sim

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

#------------------------- GCU-3.x -------------------------#
GCU3_REL_PATH=260420/gcu30
GCU3_PLATFORM_NAME=TopsPlatform_1.8.20260319.113917_tmp-20260319113917-32db2a_deb_amd64.run
GCU3_PLATFORM_PKG=$(TOOLCHAIN_DIR)/$(GCU3_PLATFORM_NAME)
GCU3_PLATFORM_PKG_MD5:=bf1ac35b2d6d05cfe0f1254597c4fca9
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
	$(GCU3_PLATFORM_PKG) -y -C topsruntime,topscc,topsprof --install-dir $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/opt/tops/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/opt;

# specific used for package testing
install-gcu3-topscc: check-gcu3-platform-kit
	sudo $(GCU3_PLATFORM_PKG) -y -C topsruntime,topscc

setup-gcu3-platform-kit: check-gcu3-platform-kit
	@if [ "$(CUR_GCU3_PLATFORM_PKG_MD5)" != "$(GCU3_PLATFORM_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu3-platform-kit; \
	fi

#------------------------- GCU-4.x -------------------------#
GCU4_REL_PATH=251103/gcu400
GCU4_PLATFORM_NAME=TopsPlatform_1.6.1.216-3aea03_deb_amd64.run
GCU4_PLATFORM_PKG=$(TOOLCHAIN_DIR)/$(GCU4_PLATFORM_NAME)
GCU4_PLATFORM_PKG_MD5:=c42fccec6ec4d5b472ed91e51d0c4772
CUR_GCU4_PLATFORM_PKG_MD5:=$(shell md5sum $(GCU4_PLATFORM_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu4-platform-kit:
	@if [ "$(CUR_GCU4_PLATFORM_PKG_MD5)" != "$(GCU4_PLATFORM_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu4-platform-kit; \
	else \
		echo "$(GCU4_PLATFORM_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu4-platform-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU4_REL_PATH)/$(GCU4_PLATFORM_NAME) -o $(GCU4_PLATFORM_PKG);\
	chmod +x $(GCU4_PLATFORM_PKG);

install-gcu4-platform-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	$(GCU4_PLATFORM_PKG) -y -C topsruntime,topscc --install-dir $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/opt/tops/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/opt;

install-gcu4-platform-sim-kit:
	mkdir -p $(GCU_SIM_DIR) ;\
	$(GCU4_PLATFORM_PKG) -y -C topsruntime,topscc,topspti --install-dir $(GCU_SIM_DIR); \
	rsync -av $(GCU_SIM_DIR)/opt/tops/* $(GCU_SIM_DIR); \
	rm -fr $(GCU_SIM_DIR)/opt;

setup-gcu4-platform-kit: check-gcu4-platform-kit
	@if [ "$(CUR_GCU4_PLATFORM_PKG_MD5)" != "$(GCU4_PLATFORM_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu4-platform-kit; \
	fi

setup-gcu4-platform-sim-kit: check-gcu4-platform-kit
	@if [ "$(CUR_GCU4_PLATFORM_PKG_MD5)" != "$(GCU4_PLATFORM_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu4-platform-sim-kit; \
	fi

install-gcu-sim:
	mkdir -p $(GCU_SIM_DIR)/lib
	@if [ ! -f "$(GCU_SIM_DIR)/lib/libgcusim.so" ]; then \
		wget http://artifact.enflame.cn/artifactory/release_center/GCUSIM/1.0.91/libgcusim.so; \
		mv libgcusim.so $(GCU_SIM_DIR)/lib/libgcusim.so; \
	fi;

#------------------------- GCU-5.x -------------------------#
GCU5_REL_PATH=251103/gcu500
GCU5_TOPSCC_RT_NAME=topscc_rt.1.6.20251029.tar.gz
GCU5_TOPSCC_RT=$(TOOLCHAIN_DIR)/$(GCU5_PLATFORM_NAME)
GCU5_TOPSCC_RT_MD5:=bfc0034a433662db7ee3c6a3ac90d190
CUR_GCU5_TOPSCC_RT_MD5:=$(shell md5sum $(GCU5_TOPSCC_RT) 2>/dev/null| cut -d ' ' -f 1)

check-gcu5-tops-kit:
	@if [ "$(CUR_GCU5_TOPSCC_RT_MD5)" != "$(GCU5_TOPSCC_RT_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu5-tops-kit; \
	else \
		echo "$(GCU5_TOPSCC_RT) MD5 hash matches. No need to download."; \
	fi

download-gcu5-tops-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU5_REL_PATH)/$(GCU5_TOPSCC_RT_NAME) -o $(GCU5_TOPSCC_RT_NAME)

install-gcu5-tops-kit:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	tar -zvxf $(GCU5_TOPSCC_RT_NAME) -C $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/topscc_rt/opt/tops/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/topscc_rt;

setup-gcu5-tops-kit: check-gcu5-tops-kit
	@if [ "$(CUR_GCU5_TOPSCC_RT_MD5)" != "$(GCU5_TOPSCC_RT_MD5)"  ]; then \
	    $(MAKE) install-gcu5-tops-kit; \
	fi

install-gcu-sim5:
	mkdir -p $(GCU_SIM_DIR)/lib
	@if [ ! -f "$(GCU_SIM_DIR)/lib/libgcusim5.so" ]; then \
		wget http://artifact.enflame.cn:80/artifactory/release_center/GCUSIM/5.0.24/libgcusim5.so; \
		mv libgcusim5.so $(GCU_SIM_DIR)/lib/libgcusim5.so; \
	fi;

#-----------------------------------------------------------#
ACORE_REL_PATH=260420/gcu30
ACORE_NAME=topsacore_3.3.20260414-e362563.tar.gz
ACORE_PKG=$(TOOLCHAIN_DIR)/$(ACORE_NAME)
ACORE_PKG_MD5=525b4b6cddbc21b8e8ef376418e641d9
CUR_ACORE_PKG_MD5:=$(shell md5sum $(ACORE_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-acore:
	@if [ "$(ACORE_PKG_MD5)" != "$(CUR_ACORE_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the acore package..."; \
		$(MAKE) download-acore; \
	else \
		echo "$(ACORE_PKG) MD5 hash matches. No need to download."; \
	fi

download-acore:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(ACORE_REL_PATH)/$(ACORE_NAME) -o $(ACORE_PKG)

install-acore:
	tar -zvxf $(TOOLCHAIN_DIR)/$(ACORE_NAME) -C $(TOOLCHAIN_DIR)/

setup-acore: check-acore
	@if [ "$(CUR_ACORE_PKG_MD5)" != "$(ACORE_PKG_MD5)"  ]; then \
	    $(MAKE) install-acore; \
	fi

#-----------------------------------------------------------#
CUDA_NAME=cuda_11.3.r11.3.tgz
CUDA_URL=ftp://172.16.11.18/\%2fdev/choreo-toolchain/
CUDA_PKG=$(TOOLCHAIN_DIR)/$(CUDA_NAME)
CUDA_PKG_MD5=c623a27f7aef41ec3aa134e1e61b9fa7
CUR_CUDA_PKG_MD5:=$(shell md5sum $(CUDA_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-cuda:
	@if [ "$(CUDA_PKG_MD5)" != "$(CUR_CUDA_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the cuda package..."; \
		$(MAKE) download-cuda; \
	else \
		echo "$(CUDA_PKG) MD5 hash matches. No need to download."; \
	fi

download-cuda:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(CUDA_NAME) -o $(CUDA_NAME);\

install-cuda:
	tar -zvxf $(TOOLCHAIN_DIR)/$(CUDA_NAME) -C $(TOOLCHAIN_DIR)/

setup-cuda: check-cuda
	@if [ "$(CUR_CUDA_PKG_MD5)" != "$(CUDA_PKG_MD5)"  ]; then \
	    $(MAKE) install-cuda; \
	fi


gcu2-kmd: check-gcu-platform-kit
	sudo $(GCU_PLATFORM_PKG) -y -C enflame

# --force-asic is applied for some specific environment
gcu3-kmd: check-gcu3-platform-kit
	sudo $(GCU3_PLATFORM_PKG) -y -C enflame --force-asic

gcu4-kmd: check-gcu4-platform-kit
	sudo $(GCU4_PLATFORM_PKG) -y -C enflame --force-asic


gcu2-kit: setup-gcu-sfc-kit setup-gcu-platform-kit
gcu3-kit: setup-gcu3-platform-kit
gcu4-kit: setup-gcu4-platform-kit install-gcu-sim
gcu4-sim-kit: setup-gcu4-platform-sim-kit install-gcu-sim
gcu-sim4: install-gcu-sim
gcu5-kit: setup-gcu5-tops-kit install-gcu-sim5
gcu5-sim-kit: setup-gcu5-tops-kit install-gcu-sim5
gcu-sim5: install-gcu-sim5

gcu2-install: install-gcu-sfc-kit install-gcu-platform-kit
gcu3-install: install-gcu3-platform-kit
gcu4-install: install-gcu4-platform-kit
