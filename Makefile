
NVCC = nvcc
TARGET = build/benchmark_all
SOURCE = src/benchmark_all.cu

NVCC_FLAGS = -O2 -std=c++17
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	mkdir -p build
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET) $(OPENCV_FLAGS)

run: all
	./$(TARGET)

clean:
	rm -rf build
