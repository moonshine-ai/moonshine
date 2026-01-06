#include "resampler.h"

#include "debug-utils.h"

const std::vector<float> resample_audio(const std::vector<float> &audio,
                                        float input_sample_rate,
                                        float output_sample_rate) {
  if (input_sample_rate == output_sample_rate) {
    return audio;
  }
  if (input_sample_rate > output_sample_rate) {
    return downsample_audio(audio, input_sample_rate, output_sample_rate);
  }
  return upsample_audio(audio, input_sample_rate, output_sample_rate);
}

const std::vector<float> downsample_audio(const std::vector<float> &audio,
                                          float input_sample_rate,
                                          float output_sample_rate) {
  const size_t input_audio_size = audio.size();
  size_t output_audio_size =
      input_audio_size * output_sample_rate / input_sample_rate;
  std::vector<float> output_audio(output_audio_size);

  const float ratio = input_sample_rate / output_sample_rate;

  for (size_t i = 0; i < output_audio_size; i++) {
    // Calculate the range of input samples that contribute to this output
    // sample
    float start_pos = i * ratio;
    float end_pos = (i + 1) * ratio;

    size_t start_idx = static_cast<size_t>(start_pos);
    size_t end_idx = static_cast<size_t>(end_pos);

    // Ensure we don't go out of bounds
    if (end_idx >= input_audio_size) {
      end_idx = input_audio_size - 1;
    }

    // Box sampling: average all samples in the range
    float sum = 0.0f;
    size_t count = 0;

    for (size_t j = start_idx; j <= end_idx; j++) {
      sum += audio[j];
      count++;
    }

    output_audio[i] = (count > 0) ? (sum / count) : 0.0f;
  }

  return output_audio;
}

const std::vector<float> upsample_audio(const std::vector<float> &audio,
                                        float input_sample_rate,
                                        float output_sample_rate) {
  const size_t input_audio_size = audio.size();
  size_t output_audio_size =
      input_audio_size * output_sample_rate / input_sample_rate;
  std::vector<float> output_audio(output_audio_size);

  const float ratio = input_sample_rate / output_sample_rate;

  for (size_t i = 0; i < output_audio_size; i++) {
    // Calculate the exact position in the input array
    float pos = i * ratio;

    // Get the integer part (index) and fractional part
    size_t index = static_cast<size_t>(pos);
    float fraction = pos - index;

    // Handle boundary cases
    if (index >= input_audio_size - 1) {
      // If we're at or beyond the last sample, just use the last sample
      output_audio[i] = audio[input_audio_size - 1];
    } else {
      // Bilinear (linear) interpolation between two adjacent samples
      float sample0 = audio[index];
      float sample1 = audio[index + 1];
      output_audio[i] = sample0 + fraction * (sample1 - sample0);
    }
  }

  return output_audio;
}