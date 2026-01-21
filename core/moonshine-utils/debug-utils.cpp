#include "debug-utils.h"

#include <fcntl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

bool load_wav_data(const char *path, float **out_float_data,
                   size_t *out_num_samples, int32_t *out_sample_rate) {
  *out_float_data = nullptr;
  *out_num_samples = 0;

  // Open the file in binary mode
  FILE *file = std::fopen(path, "rb");
  if (!file) {
    std::perror("Failed to open WAV file");
    return false;
  }

  // Read the RIFF header
  char riff_header[4];
  if (std::fread(riff_header, 1, 4, file) != 4 ||
      std::strncmp(riff_header, "RIFF", 4) != 0) {
    std::fclose(file);
    std::fprintf(stderr, "Not a RIFF file\n");
    return false;
  }

  // Skip chunk size and check WAVE
  std::fseek(file, 4, SEEK_CUR);
  char wave_header[4];
  if (std::fread(wave_header, 1, 4, file) != 4 ||
      std::strncmp(wave_header, "WAVE", 4) != 0) {
    std::fclose(file);
    std::fprintf(stderr, "Not a WAVE file\n");
    return false;
  }

  // Find the "fmt " chunk
  char chunk_id[4];
  uint32_t chunk_size = 0;
  bool found_fmt = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
      found_fmt = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_fmt) {
    std::fclose(file);
    std::fprintf(stderr, "No fmt chunk found\n");
    return false;
  }

  // Read fmt chunk
  uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0, byte_rate = 0;
  uint16_t block_align = 0;
  if (chunk_size < 16) {
    std::fclose(file);
    std::fprintf(stderr, "fmt chunk too small\n");
    return false;
  }
  std::fread(&audio_format, sizeof(uint16_t), 1, file);
  std::fread(&num_channels, sizeof(uint16_t), 1, file);
  std::fread(&sample_rate, sizeof(uint32_t), 1, file);
  std::fread(&byte_rate, sizeof(uint32_t), 1, file);
  std::fread(&block_align, sizeof(uint16_t), 1, file);
  std::fread(&bits_per_sample, sizeof(uint16_t), 1, file);
  // Skip any extra fmt bytes
  if (chunk_size > 16) std::fseek(file, chunk_size - 16, SEEK_CUR);

  if (audio_format != 1 || bits_per_sample != 16) {
    std::fclose(file);
    std::fprintf(stderr, "Only 16-bit PCM WAV files are supported\n");
    return false;
  }

  // Find the "data" chunk
  bool found_data = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "data", 4) == 0) {
      found_data = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_data) {
    std::fclose(file);
    std::fprintf(stderr, "No data chunk found\n");
    return false;
  }

  // Read PCM data
  size_t num_samples = chunk_size / (bits_per_sample / 8);
  if (num_samples == 0) {
    std::fclose(file);
    std::fprintf(stderr, "No samples found\n");
    return false;
  }
  float *result_data = (float *)malloc(num_samples * sizeof(float));
  for (size_t i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    if (std::fread(&sample, sizeof(int16_t), 1, file) != 1) {
      std::fclose(file);
      std::fprintf(stderr, "Failed to read sample %zu\n", i);
      return false;
    }
    result_data[i] = static_cast<float>(sample) / 32768.0f;
  }
  std::fclose(file);
  *out_float_data = result_data;
  *out_num_samples = num_samples;
  *out_sample_rate = sample_rate;
  return true;
}

bool save_wav_data(const char *path, const float *audio_data,
                   size_t num_samples, uint32_t sample_rate) {
  FILE *file = std::fopen(path, "wb");
  if (!file) {
    std::perror("Failed to open WAV file");
    return false;
  }

  std::vector<int16_t> audio_int16(num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    const float float_value = gate(audio_data[i], -1.0f, 1.0f);
    audio_int16[i] = static_cast<int16_t>(float_value * 32768.0f);
  }

  // Write the RIFF header
  char riff_header[4] = {'R', 'I', 'F', 'F'};
  std::fwrite(riff_header, 1, 4, file);
  // Write the chunk size
  uint32_t chunk_size = 36 + (uint32_t)num_samples * 2;
  std::fwrite(&chunk_size, 4, 1, file);
  // Write the format
  char format[4] = {'W', 'A', 'V', 'E'};
  std::fwrite(format, 1, 4, file);
  // Write the fmt chunk
  char fmt_chunk[4] = {'f', 'm', 't', ' '};
  std::fwrite(fmt_chunk, 1, 4, file);
  // Write the fmt chunk size
  uint32_t fmt_chunk_size = 16;
  std::fwrite(&fmt_chunk_size, 4, 1, file);
  // Write the fmt chunk data
  uint16_t audio_format = 1;
  std::fwrite(&audio_format, 2, 1, file);
  uint16_t num_channels = 1;
  std::fwrite(&num_channels, 2, 1, file);
  std::fwrite(&sample_rate, 4, 1, file);
  uint32_t byte_rate = sample_rate * 2;
  std::fwrite(&byte_rate, 4, 1, file);
  uint16_t block_align = 2;
  std::fwrite(&block_align, 2, 1, file);
  uint16_t bits_per_sample = 16;
  std::fwrite(&bits_per_sample, 2, 1, file);
  // Write the data chunk
  char data_chunk[4] = {'d', 'a', 't', 'a'};
  std::fwrite(data_chunk, 1, 4, file);
  // Write the data chunk size
  uint32_t data_chunk_size = (uint32_t)num_samples * 2;
  std::fwrite(&data_chunk_size, 4, 1, file);
  // Write the data
  std::fwrite(audio_int16.data(), 2, num_samples, file);
  std::fclose(file);
  return true;
}

std::string float_vector_stats_to_string(const std::vector<float> &vector) {
  std::string result =
      "float_vector_stats(size=" + std::to_string(vector.size()) + ", ";
  float min = *std::min_element(vector.begin(), vector.end());
  result += "min=" + std::to_string(min) + ", ";
  float max = *std::max_element(vector.begin(), vector.end());
  result += "max=" + std::to_string(max) + ", ";
  float mean =
      std::accumulate(vector.begin(), vector.end(), 0.0f) / vector.size();
  result += "mean=" + std::to_string(mean) + ", ";
  float std = std::sqrt(std::accumulate(vector.begin(), vector.end(), 0.0f,
                                        [mean](float acc, float x) {
                                          return acc + (x - mean) * (x - mean);
                                        }) /
                        vector.size());
  result += "std=" + std::to_string(std) + ")";
  return result;
}

std::vector<uint8_t> load_file_into_memory(const std::string &path) {
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) {
    THROW_WITH_LOG(("Failed to open file: '" + path + "'").c_str());
  }
  std::fseek(file, 0, SEEK_END);
  size_t size = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  std::vector<uint8_t> data(size);
  size_t bytes_read = std::fread(data.data(), 1, size, file);
  if (bytes_read != size) {
    THROW_WITH_LOG(("Failed to read file: '" + path +
                    "' completely. Expected " + std::to_string(size) +
                    " bytes, but read " + std::to_string(bytes_read) +
                    " bytes.")
                       .c_str());
  }
  std::fclose(file);
  return data;
}

void save_memory_to_file(const std::string &path,
                         const std::vector<uint8_t> &data) {
  FILE *file = std::fopen(path.c_str(), "wb");
  if (!file) {
    THROW_WITH_LOG(("Failed to open file: '" + path + "'").c_str());
  }
  size_t bytes_written = std::fwrite(data.data(), 1, data.size(), file);
  if (bytes_written != data.size()) {
    THROW_WITH_LOG(("Failed to write file: '" + path +
                    "' completely. Expected " + std::to_string(data.size()) +
                    " bytes, but wrote " + std::to_string(bytes_written) +
                    " bytes.")
                       .c_str());
  }
  std::fclose(file);
}