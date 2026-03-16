#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "moonshine-c-api.h"

// Simple test: transcribe with word_timestamps=true and verify output
int main(int argc, char **argv) {
  // Default paths, can be overridden by command line args
  const char *model_path = "../../test-assets/tiny-en";
  const char *wav_path = "../../test-assets/beckett.wav";
  if (argc > 1) model_path = argv[1];
  if (argc > 2) wav_path = argv[2];

  // Load with word_timestamps enabled
  struct transcriber_option_t options[] = {
    {.name = "word_timestamps", .value = "true"},
    {.name = "identify_speakers", .value = "false"},
  };

  printf("Loading model from %s with word_timestamps=true...\n", model_path);
  int32_t handle = moonshine_load_transcriber_from_files(
    model_path, MOONSHINE_MODEL_ARCH_TINY, options, 2,
    moonshine_get_version());

  if (handle < 0) {
  printf("FAIL: Failed to load transcriber: %s\n",
       moonshine_error_to_string(handle));
  return 1;
  }
  printf("Model loaded (handle=%d)\n\n", handle);

  // Load WAV file
  FILE *f = fopen(wav_path, "rb");
  if (!f) {
  printf("FAIL: Cannot open %s\n", wav_path);
  return 1;
  }

  // Read WAV header (44 bytes) then raw PCM data
  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 44, SEEK_SET);  // Skip WAV header
  long data_size = file_size - 44;
  int16_t *raw = (int16_t *)malloc(data_size);
  fread(raw, 1, data_size, f);
  fclose(f);

  // Convert to float
  long num_samples = data_size / sizeof(int16_t);
  float *audio = (float *)malloc(num_samples * sizeof(float));
  for (long i = 0; i < num_samples; i++) {
  audio[i] = raw[i] / 32768.0f;
  }
  free(raw);

  float audio_duration = num_samples / 16000.0f;
  printf("Audio: %s (%.2fs, %ld samples)\n\n", wav_path, audio_duration,
     num_samples);

  // Transcribe
  struct transcript_t *transcript = NULL;
  int32_t err = moonshine_transcribe_without_streaming(handle, audio,
                             num_samples, 16000, 0,
                             &transcript);
  free(audio);

  if (err != 0) {
  printf("FAIL: Transcription failed: %s\n", moonshine_error_to_string(err));
  moonshine_free_transcriber(handle);
  return 1;
  }

  if (!transcript || transcript->line_count == 0) {
  printf("FAIL: No transcript lines produced\n");
  moonshine_free_transcriber(handle);
  return 1;
  }

  printf("Transcript lines: %llu\n", transcript->line_count);

  int total_words = 0;
  int monotonicity_violations = 0;

  for (uint64_t i = 0; i < transcript->line_count; i++) {
  struct transcript_line_t *line = &transcript->lines[i];
  printf("\nLine %llu: \"%s\"\n", i, line->text ? line->text : "<null>");
  printf("  start_time=%.3f, duration=%.3f\n", line->start_time,
       line->duration);
  printf("  word_count=%llu\n", line->word_count);

  if (line->words && line->word_count > 0) {
    float prev_start = -1.0f;
    for (uint64_t j = 0; j < line->word_count; j++) {
    const struct transcript_word_t *word = &line->words[j];
    printf("    [%7.3fs - %7.3fs] %-15s  (conf: %.2f)\n", word->start,
         word->end, word->text ? word->text : "<null>", word->confidence);

    if (word->start < prev_start) {
      monotonicity_violations++;
    }
    prev_start = word->start;
    total_words++;
    }
  }
  }

  printf("\n=== Results ===\n");
  printf("Total words: %d\n", total_words);
  printf("Monotonicity violations: %d\n", monotonicity_violations);

  if (total_words == 0) {
  printf("FAIL: No words produced\n");
  moonshine_free_transcriber(handle);
  return 1;
  }

  if (monotonicity_violations > 0) {
  printf("FAIL: Monotonicity violations detected\n");
  moonshine_free_transcriber(handle);
  return 1;
  }

  printf("PASS: %d words with correct monotonic ordering\n", total_words);

  moonshine_free_transcriber(handle);
  return 0;
}
