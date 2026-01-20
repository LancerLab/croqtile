# Variables
CXX = g++
NV_CXX = nvcc
CXXFLAGS = -Wall -g -std=c++17
CUDA_SYS_INCLUDES="-I/usr/local/cuda/include"
NV_FLAGS = -gencode arch=compute_86,code=sm_86 -rdc=true -lcublas $(CUDA_SYS_INCLUDES) --expt-relaxed-constexpr -static-libstdc++

ROOT=../../
GTEST_DIR = $(ROOT)/extern/gtest/
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a
HEADER_FILES :=  $(shell find $(ROOT)/lib -name '*.hpp') $(ROOT)/runtime/choreo.h
CHOREO_HEADERS :=  $(shell find $(ROOT)/lib -name '*.hpp')

# Your application's files
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGETS = $(SOURCES:.cpp=.test)

NV_SOURCES = $(wildcard *.cu)
NV_TARGETS = $(NV_SOURCES:.cu=.cutest)
NV_TARGETS =

# Main target
all: $(TARGETS) $(NV_TARGETS)

# Google Test build targets
$(GTEST_DIR)/libgtest.a: $(GTEST_DIR)/googletest/src/gtest-all.cc
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR)/googletest/include -I$(GTEST_DIR)/googletest -pthread -c $< -o $(GTEST_DIR)/gtest-all.o
	ar -rv $@ $(GTEST_DIR)/gtest-all.o

$(GTEST_DIR)/libgtest_main.a: $(GTEST_DIR)/googletest/src/gtest_main.cc
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR)/googletest/include -I$(GTEST_DIR)/googletest -pthread -c $< -o $(GTEST_DIR)/gtest_main.o
	ar -rv $@ $(GTEST_DIR)/gtest_main.o

target_registry.o: $(ROOT)/lib/target_registry.cpp
	$(CXX) $(CXXFLAGS) -I$(ROOT)/lib -c -o $@ $< -pthread -static-libstdc++

%.test: %.o $(GTEST_LIBS) target_registry.o
	$(CXX) $(CXXFLAGS) -I$(ROOT)/ -I$(ROOT)/lib -L$(GTEST_DIR) -lgtest_main -lgtest -o $@ $< target_registry.o -pthread -static-libstdc++

# Pattern rule for object files
%.o: %.cpp $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) -I$(ROOT)/ -I$(ROOT)/lib -I$(GTEST_DIR)/googletest/include/ -c $< -o $@

%.cutest: %.cu $(GTEST_LIBS) target_registry.o
	$(NV_CXX) ${NV_FLAGS} -I$(ROOT)/ -I$(ROOT)/lib -I$(GTEST_DIR)/googletest/include/ -L../../ -L$(GTEST_DIR) target_registry.o -lgtest_main -lgtest -o $@ $< -lpthread

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS) $(NV_TARGET) $(TARGETS) $(GTEST_DIR)/*.o $(GTEST_LIBS)

# Additional rules
test: $(TARGETS) $(NV_TARGETS)
	@for test in $(TARGETS); do \
		echo "Running $$test..."; \
		./$$test || exit $$?; \
	done
	@for nv_test in $(NV_TARGETS); do \
		echo "Running $$nv_test..."; \
		./$$nv_test || exit $$?; \
	done

.PHONY: all clean test
