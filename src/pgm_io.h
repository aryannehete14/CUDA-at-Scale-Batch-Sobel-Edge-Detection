#ifndef PGM_IO_H_
#define PGM_IO_H_

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pgm {

struct Image {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> pixels;  // row-major, one byte per pixel
};

// Reads a binary PGM (P5) file. Throws std::runtime_error on failure.
inline Image ReadPgm(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file: " + path);
  }

  std::string magic;
  file >> magic;
  if (magic != "P5") {
    throw std::runtime_error("Unsupported PGM format (need P5): " + path);
  }

  auto skip_comments = [&file]() {
    while (file.peek() == '#') {
      std::string line;
      std::getline(file, line);
    }
  };

  int width = 0, height = 0, max_val = 0;
  skip_comments();
  file >> width;
  skip_comments();
  file >> height;
  skip_comments();
  file >> max_val;
  file.get();  // consume single whitespace char after header

  if (width <= 0 || height <= 0 || max_val > 255) {
    throw std::runtime_error("Invalid PGM header in: " + path);
  }

  Image img;
  img.width = width;
  img.height = height;
  img.pixels.resize(static_cast<size_t>(width) * height);
  file.read(reinterpret_cast<char*>(img.pixels.data()), img.pixels.size());
  if (!file) {
    throw std::runtime_error("Unexpected EOF reading pixel data: " + path);
  }
  return img;
}

// Writes a binary PGM (P5) file.
inline void WritePgm(const std::string& path, const Image& img) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open output file: " + path);
  }
  file << "P5\n" << img.width << " " << img.height << "\n255\n";
  file.write(reinterpret_cast<const char*>(img.pixels.data()),
             img.pixels.size());
}

}  // namespace pgm

#endif  // PGM_IO_H_