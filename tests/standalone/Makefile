# Variables
CXX = g++
CXXFLAGS = -Wall -g -std=c++17
GTEST_DIR = ../../extern/gtest/
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a
HEADER_FILES :=  $(shell find ../.. -name '*.hpp')

# Your application's files
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGETS = $(SOURCES:.cpp=.test)

# Main target
all: $(TARGETS)

# Google Test build targets
$(GTEST_DIR)/libgtest.a: $(GTEST_DIR)/googletest/src/gtest-all.cc
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR)/googletest/include -I$(GTEST_DIR)/googletest -pthread -c $< -o $(GTEST_DIR)/gtest-all.o
	ar -rv $@ $(GTEST_DIR)/gtest-all.o

$(GTEST_DIR)/libgtest_main.a: $(GTEST_DIR)/googletest/src/gtest_main.cc
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR)/googletest/include -I$(GTEST_DIR)/googletest -pthread -c $< -o $(GTEST_DIR)/gtest_main.o
	ar -rv $@ $(GTEST_DIR)/gtest_main.o

%.test: %.o $(GTEST_LIBS)
	$(CXX) $(CXXFLAGS) -L../../ -L$(GTEST_DIR) -lgtest_main -lgtest -o $@ $< -pthread

# Pattern rule for object files
%.o: %.cpp $(HEADER_FILES)
	$(CXX) $(CXXFLAGS) -I../../ -I$(GTEST_DIR)/googletest/include/ -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS) $(TARGETS) $(GTEST_DIR)/*.o $(GTEST_LIBS)

# Additional rules
test: $(TARGETS)
	@for test in $(TARGETS); do ./$${test}; done;

.PHONY: all clean test
