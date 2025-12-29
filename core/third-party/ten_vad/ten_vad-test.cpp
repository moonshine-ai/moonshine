#include "ten_vad.h"

#include <cstdio>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("ten_vad") {
    SUBCASE("ten_vad_create") {
        ten_vad_handle_t handle;
        CHECK(ten_vad_create(&handle, 256, 0.5) == 0);
        CHECK(ten_vad_destroy(&handle) == 0);
    }
    SUBCASE("ten_vad_process") {
        ten_vad_handle_t handle;
        CHECK(ten_vad_create(&handle, 256, 0.5) == 0);
        float probability;
        int flag;
        int16_t audio_data[256];
        int audio_data_length = 256;
        CHECK(ten_vad_process(handle, audio_data, audio_data_length, &probability, &flag) == 0);
        CHECK(ten_vad_destroy(&handle) == 0);
    }
}