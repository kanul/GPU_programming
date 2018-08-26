CXX = nvcc

# path #
SRC_PATH = src
BUILD_PATH = obj

# executable # 
BIN_NAME = rbm-mnist

# extensions #
SRC_EXT = cpp
SRC_EXT2 = cu

CUDA_PATH ?= "/usr/local/cuda-8.0"

# code lists #
# Find all source files in the source directory, sorted by
# most recently modified
SOURCES1 = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
SOURCES2 = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT2)' | sort -k 1nr | cut -f2-)
# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS1 = $(SOURCES1:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
OBJECTS2 = $(SOURCES2:$(SRC_PATH)/%.$(SRC_EXT2)=$(BUILD_PATH)/%.cu.o)
# Set the dependency files that will be used to add header dependencies
DEPS1 = $(OBJECTS1:.o=.d)
DEPS2 = $(OBJECTS2:.o=.d)

HIGHEST_SM = 52
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

# flags #
#COMPILE_FLAGS = -std=c++11 -Wall -Wextra -g -O2
COMPILE_FLAGS = -std=c++11 -g -O2 -m64 $(GENCODE_FLAGS)
INCLUDES = -I include/ -I /usr/local/include -I../../common/inc -I $(CUDA_PATH)/include
# Space-separated pkg-config libraries used by this project
LIBS = -lpthread -lz -L$(CUDA_PATH)/lib64 -lcudart -lcuda 

.PHONY: default_target
default_target: release

.PHONY: release
release: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS)
release: dirs
	@$(MAKE) all

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS1))

.PHONY: clean
clean:
	@echo "Deleting $(BIN_NAME) symlink"
	@$(RM) $(BIN_NAME)
	@echo "Deleting directories"
	@$(RM) -r $(BUILD_PATH)

# checks the executable and symlinks to the output
.PHONY: all
all: $(BIN_NAME)

# Creation of the executable
$(BIN_NAME): $(OBJECTS1) $(OBJECTS2)
	@echo "Linking: $@"
	$(CXX) $(OBJECTS1) $(OBJECTS2) $(LIBS) -o $@

# Add dependency files, if they exist
#-include $(DEPS) $(DEPS2)

# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	#$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@
$(BUILD_PATH)/%.cu.o: $(SRC_PATH)/%.$(SRC_EXT2)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
