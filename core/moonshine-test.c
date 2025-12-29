#include "moonshine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(ANDROID)
#include <android/log.h>
#define LOG(...) __android_log_print(ANDROID_LOG_WARN, "Native", __VA_ARGS__)
#else
#define LOG(...) fprintf(stderr, __VA_ARGS__)
#endif

static int load_wav_data(const char *path, float **out_float_data,
                         size_t *out_num_samples);

int main(int argc, char *argv[]) {
  const char *wav_path = "beckett.wav";
  if (argc >= 2) {
    wav_path = argv[1];
  }

  int32_t model_type = MOONSHINE_MODEL_TYPE_BASE;
  if (argc >= 3) {
    model_type = atoi(argv[2]);
  }

  const char *encoder_model_path = "encoder_model.ort";
  if (argc >= 4) {
    encoder_model_path = argv[3];
  }
  const char *decoder_model_path = "decoder_model_merged.ort";
  if (argc >= 5) {
    decoder_model_path = argv[4];
  }

  const char *tokenizer_path = "tokenizer.bin";
  if (argc >= 6) {
    tokenizer_path = argv[5];
  }

  float *wav_data = 0;
  size_t wav_data_size = 0;
  if (load_wav_data(wav_path, &wav_data, &wav_data_size) != 0) {
    LOG("Failed to load WAV file: '%s'\n", wav_path);
    return 1;
  }

  LOG("Loading model from: '%s', '%s', '%s', %d\n", encoder_model_path,
      decoder_model_path, tokenizer_path, model_type);

  moonshine_handle_t model = moonshine_load_model(
      encoder_model_path, decoder_model_path, tokenizer_path, model_type);

  if (model == -1) {
    LOG("Failed to load model: '%s', '%s', '%s', %d\n", encoder_model_path,
        decoder_model_path, tokenizer_path, model_type);
    return 1;
  }

  char *out_text = NULL;
  const int transcribe_error =
      moonshine_transcribe(model, wav_data, wav_data_size, &out_text);
  if (transcribe_error != 0) {
    LOG("Failed to transcribe WAV file: '%s'\n", wav_path);
    return 1;
  }
  printf("%s\n", out_text);

  moonshine_free_model(model);
  free(wav_data);
  wav_data = 0;

  return 0;
}

int load_wav_data(const char *path, float **out_float_data,
                  size_t *out_num_samples) {
  *out_float_data = 0;
  *out_num_samples = 0;

  // Open the file in binary mode
  FILE *file = fopen(path, "rb");
  if (!file) {
    perror("Failed to open WAV file");
    return 1;
  }

  // Read the RIFF header
  char riff_header[4];
  if (fread(riff_header, 1, 4, file) != 4 ||
      strncmp(riff_header, "RIFF", 4) != 0) {
    fclose(file);
    fprintf(stderr, "Not a RIFF file\n");
    return 1;
  }

  // Skip chunk size and check WAVE
  fseek(file, 4, SEEK_CUR);
  char wave_header[4];
  if (fread(wave_header, 1, 4, file) != 4 ||
      strncmp(wave_header, "WAVE", 4) != 0) {
    fclose(file);
    fprintf(stderr, "Not a WAVE file\n");
    return 1;
  }

  // Find the "fmt " chunk
  char chunk_id[4];
  uint32_t chunk_size = 0;
  int found_fmt = 0;
  while (fread(chunk_id, 1, 4, file) == 4) {
    if (fread(&chunk_size, 4, 1, file) != 1)
      break;
    if (strncmp(chunk_id, "fmt ", 4) == 0) {
      found_fmt = 1;
      break;
    }
    fseek(file, chunk_size, SEEK_CUR);
  }
  if (found_fmt == 0) {
    fclose(file);
    fprintf(stderr, "No fmt chunk found\n");
    return 1;
  }

  // Read fmt chunk
  uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0, byte_rate = 0;
  uint16_t block_align = 0;
  if (chunk_size < 16) {
    fclose(file);
    fprintf(stderr, "fmt chunk too small\n");
    return 1;
  }
  fread(&audio_format, sizeof(uint16_t), 1, file);
  fread(&num_channels, sizeof(uint16_t), 1, file);
  fread(&sample_rate, sizeof(uint32_t), 1, file);
  fread(&byte_rate, sizeof(uint32_t), 1, file);
  fread(&block_align, sizeof(uint16_t), 1, file);
  fread(&bits_per_sample, sizeof(uint16_t), 1, file);
  // Skip any extra fmt bytes
  if (chunk_size > 16)
    fseek(file, chunk_size - 16, SEEK_CUR);

  if (audio_format != 1 || bits_per_sample != 16) {
    fclose(file);
    fprintf(stderr, "Only 16-bit PCM WAV files are supported\n");
    return 1;
  }

  // Find the "data" chunk
  int found_data = 0;
  while (fread(chunk_id, 1, 4, file) == 4) {
    if (fread(&chunk_size, 4, 1, file) != 1)
      break;
    if (strncmp(chunk_id, "data", 4) == 0) {
      found_data = 1;
      break;
    }
    fseek(file, chunk_size, SEEK_CUR);
  }
  if (found_data == 0) {
    fclose(file);
    fprintf(stderr, "No data chunk found\n");
    return 1;
  }

  // Read PCM data
  size_t num_samples = chunk_size / (bits_per_sample / 8);
  if (num_samples == 0) {
    fclose(file);
    fprintf(stderr, "No samples found\n");
    return 1;
  }
  float *result_data = (float *)malloc(num_samples * sizeof(float));
  for (size_t i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    if (fread(&sample, sizeof(int16_t), 1, file) != 1) {
      num_samples = i;
      break;
    }
    result_data[i] = sample / 32768.0f;
  }
  fclose(file);
  *out_float_data = result_data;
  *out_num_samples = num_samples;
  return 0;
}
