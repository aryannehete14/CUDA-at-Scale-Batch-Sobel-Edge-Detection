# Makefile for CUDA at Scale — Batch Sobel Edge Detection

NVCC ?= nvcc
NVCC_FLAGS := -O3 -std=c++14 -arch=sm_50
SRC_DIR := src
BIN_DIR := bin
TARGET := $(BIN_DIR)/edge_detect

SOURCES := $(SRC_DIR)/main.cu $(SRC_DIR)/sobel_kernel.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) $(SRC_DIR)/pgm_io.h $(SRC_DIR)/sobel_kernel.cuh
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)

run: all
	mkdir -p output logs
	$(TARGET) --input data --output output --log logs/run.log

clean:
	rm -rf $(BIN_DIR) output/*.pgm logs/*.log