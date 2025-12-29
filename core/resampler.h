#ifndef RESAMPLER_H
#define RESAMPLER_H

#include <vector>

const std::vector<float> resample_audio(const std::vector<float> &audio,
                                        float input_sample_rate,
                                        float output_sample_rate);

const std::vector<float> downsample_audio(const std::vector<float> &audio,
                                          float input_sample_rate,
                                          float output_sample_rate);

const std::vector<float> upsample_audio(const std::vector<float> &audio,
                                        float input_sample_rate,
                                        float output_sample_rate);
#endif