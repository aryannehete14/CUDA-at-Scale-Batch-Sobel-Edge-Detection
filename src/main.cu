#include <cuda_runtime.h>
#include <dirent.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "pgm_io.h"
#include "sobel_kernel.cuh"

namespace {

struct Args {
  std::string input_dir;
  std::string output_dir;
  std::string log_path = "logs/run.log";
};

void PrintUsage(const char* prog_name) {
  std::cerr << "Usage: " << prog_name
            << " --input <dir> --output <dir> [--log <file>]\n"
            << "  --input   Directory containing .pgm images to process\n"
            << "  --output  Directory to write edge-detected .pgm images\n"
            << "  --log     Path to write the run log (default: "
               "logs/run.log)\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_value = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag << "\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--input") {
      args->input_dir = next_value("--input");
    } else if (arg == "--output") {
      args->output_dir = next_value("--output");
    } else if (arg == "--log") {
      args->log_path = next_value("--log");
    } else if (arg == "--help" || arg == "-h") {
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  return !args->input_dir.empty() && !args->output_dir.empty();
}

// Lists files ending in ".pgm" in `dir`, sorted for reproducible ordering.
std::vector<std::string> ListPgmFiles(const std::string& dir) {
  std::vector<std::string> files;
  DIR* handle = opendir(dir.c_str());
  if (handle == nullptr) {
    throw std::runtime_error("Could not open input directory: " + dir);
  }
  struct dirent* entry;
  while ((entry = readdir(handle)) != nullptr) {
    const std::string name = entry->d_name;
    if (name.size() > 4 && name.substr(name.size() - 4) == ".pgm") {
      files.push_back(name);
    }
  }
  closedir(handle);
  std::sort(files.begin(), files.end());
  return files;
}

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << "\n";                       \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    PrintUsage(argv[0]);
    return 1;
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cerr << "No CUDA-capable GPU detected.\n";
    return 1;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::vector<std::string> files;
  try {
    files = ListPgmFiles(args.input_dir);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  if (files.empty()) {
    std::cerr << "No .pgm files found in " << args.input_dir << "\n";
    return 1;
  }

  std::ofstream log(args.log_path);
  if (!log) {
    std::cerr << "Could not open log file: " << args.log_path << "\n";
    return 1;
  }

  log << "CUDA at Scale - Batch Sobel Edge Detection\n";
  log << "GPU: " << prop.name << " (compute capability " << prop.major << "."
      << prop.minor << ")\n";
  log << "Input directory:  " << args.input_dir << "\n";
  log << "Output directory: " << args.output_dir << "\n";
  log << "Images found: " << files.size() << "\n";
  log << "----------------------------------------\n";

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const auto batch_start = std::chrono::high_resolution_clock::now();
  size_t processed = 0;
  double total_kernel_ms = 0.0;

  for (const std::string& filename : files) {
    const std::string in_path = args.input_dir + "/" + filename;
    const std::string out_path = args.output_dir + "/" + filename;

    pgm::Image image;
    try {
      image = pgm::ReadPgm(in_path);
    } catch (const std::exception& e) {
      log << "SKIP  " << filename << " (" << e.what() << ")\n";
      continue;
    }

    const size_t num_bytes =
        static_cast<size_t>(image.width) * image.height;

    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, num_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, num_bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_input, image.pixels.data(), num_bytes,
                                cudaMemcpyHostToDevice, stream));

    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    CUDA_CHECK(cudaEventRecord(start_event, stream));
    RunSobelKernel(d_input, d_output, image.width, image.height, stream);
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_event, stop_event));
    total_kernel_ms += kernel_ms;

    pgm::Image out_image;
    out_image.width = image.width;
    out_image.height = image.height;
    out_image.pixels.resize(num_bytes);
    CUDA_CHECK(cudaMemcpyAsync(out_image.pixels.data(), d_output, num_bytes,
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    pgm::WritePgm(out_path, out_image);

    log << "OK    " << filename << "  (" << image.width << "x"
        << image.height << ")  kernel: " << kernel_ms << " ms\n";

    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    ++processed;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  const auto batch_end = std::chrono::high_resolution_clock::now();
  const double total_wall_ms =
      std::chrono::duration<double, std::milli>(batch_end - batch_start)
          .count();

  log << "----------------------------------------\n";
  log << "Images processed successfully: " << processed << " / "
      << files.size() << "\n";
  log << "Total GPU kernel time:  " << total_kernel_ms << " ms\n";
  log << "Total wall-clock time:  " << total_wall_ms << " ms\n";
  log.close();

  std::cout << "Done. Processed " << processed << "/" << files.size()
            << " images. See " << args.log_path << " for details.\n";
  return 0;
}