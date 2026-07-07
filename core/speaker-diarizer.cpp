#include "speaker-diarizer.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <utility>

#include "cpp-annote-engine.h"
#include "cpp-annote-streaming.h"
#include "debug-utils.h"

namespace {

// Total seconds of overlap between the spans of two turn lists.
double turn_overlap_seconds(
    const std::vector<cppannote::StreamingDiarizationTurn> &a, int32_t label_a,
    const std::vector<cppannote::StreamingDiarizationTurn> &b,
    int32_t label_b) {
  double total = 0.0;
  for (const auto &ta : a) {
    if (ta.speaker != label_a) {
      continue;
    }
    for (const auto &tb : b) {
      if (tb.speaker != label_b) {
        continue;
      }
      const double overlap =
          std::min(ta.end, tb.end) - std::max(ta.start, tb.start);
      if (overlap > 0.0) {
        total += overlap;
      }
    }
  }
  return total;
}

}  // namespace

struct SpeakerDiarizer::Impl {
  struct StreamState {
    std::unique_ptr<cppannote::StreamingDiarizationSession> session;
    // Snapshot generation the stable-ID mapping below was computed for.
    int mapped_generation = -1;
    // Turns from the last mapped snapshot, used for overlap matching when
    // the clustering relabels speakers on a later refresh.
    std::vector<cppannote::StreamingDiarizationTurn> mapped_turns;
    // Clustering label -> stable speaker ID for the last mapped snapshot.
    std::map<int32_t, uint64_t> label_to_stable_id;
    // Cached output turns for the last mapped snapshot.
    std::vector<SpeakerTurn> cached_turns;
  };

  cppannote::CppAnnoteEngine engine;
  SpeakerDiarizerOptions options;

  std::mutex mutex;
  std::map<int32_t, StreamState> streams;
  int32_t next_stream_id = 1;

  uint64_t next_stable_id = 0;
  uint32_t next_speaker_index = 0;
  std::map<uint64_t, uint32_t> speaker_index_map;

  explicit Impl(const SpeakerDiarizerOptions &options_in)
      : engine(), options(options_in) {
    // Same approach as line IDs in the transcriber: start from a random
    // 64-bit value and increment, so IDs are effectively unique.
    std::random_device rd;
    this->next_stable_id = (uint64_t)(rd()) << 32 | (uint64_t)(rd());
  }

  cppannote::StreamingDiarizationConfig session_config() const {
    cppannote::StreamingDiarizationConfig config;
    config.cluster_cadence = this->options.cluster_cadence;
    config.analyze_cadence = this->options.analyze_cadence;
    return config;
  }

  StreamState &get_stream(int32_t stream_id) {
    auto it = this->streams.find(stream_id);
    if (it == this->streams.end()) {
      throw std::runtime_error("SpeakerDiarizer: invalid stream ID " +
                               std::to_string(stream_id));
    }
    return it->second;
  }

  uint64_t allocate_stable_id() {
    const uint64_t id = this->next_stable_id++;
    this->speaker_index_map.insert({id, this->next_speaker_index++});
    return id;
  }

  // Maps the clustering labels in `snapshot` onto stable speaker IDs by
  // greedily matching each label against the labels of the previous mapped
  // snapshot, using total speech-time overlap. Unmatched labels are treated
  // as new speakers. Updates the stream's mapping state and cached turns.
  void map_snapshot_to_stable_ids(
      StreamState &state,
      const cppannote::StreamingDiarizationSnapshot &snapshot) {
    if (snapshot.refresh_generation == state.mapped_generation) {
      return;
    }

    // Collect the labels present in the new snapshot, in order of first
    // appearance so that new speakers get indices in speaking order.
    std::vector<int32_t> new_labels;
    for (const auto &turn : snapshot.turns) {
      if (std::find(new_labels.begin(), new_labels.end(), turn.speaker) ==
          new_labels.end()) {
        new_labels.push_back(turn.speaker);
      }
    }
    std::vector<int32_t> old_labels;
    for (const auto &entry : state.label_to_stable_id) {
      old_labels.push_back(entry.first);
    }

    // Greedy best-overlap matching between new and old labels.
    std::map<int32_t, uint64_t> new_mapping;
    std::vector<std::pair<double, std::pair<int32_t, int32_t>>> candidates;
    for (int32_t new_label : new_labels) {
      for (int32_t old_label : old_labels) {
        const double overlap = turn_overlap_seconds(
            snapshot.turns, new_label, state.mapped_turns, old_label);
        if (overlap > 0.0) {
          candidates.push_back({overlap, {new_label, old_label}});
        }
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    std::map<int32_t, bool> new_taken;
    std::map<int32_t, bool> old_taken;
    for (const auto &candidate : candidates) {
      const int32_t new_label = candidate.second.first;
      const int32_t old_label = candidate.second.second;
      if (new_taken.count(new_label) || old_taken.count(old_label)) {
        continue;
      }
      new_mapping[new_label] = state.label_to_stable_id.at(old_label);
      new_taken[new_label] = true;
      old_taken[old_label] = true;
    }
    for (int32_t new_label : new_labels) {
      if (!new_mapping.count(new_label)) {
        new_mapping[new_label] = this->allocate_stable_id();
      }
    }

    state.cached_turns.clear();
    state.cached_turns.reserve(snapshot.turns.size());
    for (const auto &turn : snapshot.turns) {
      SpeakerTurn out;
      out.start_time = (float)(turn.start);
      out.duration = (float)(turn.end - turn.start);
      out.speaker_id = new_mapping.at(turn.speaker);
      out.speaker_index = this->speaker_index_map.at(out.speaker_id);
      state.cached_turns.push_back(out);
    }
    state.label_to_stable_id = std::move(new_mapping);
    state.mapped_turns = snapshot.turns;
    state.mapped_generation = snapshot.refresh_generation;
  }
};

SpeakerDiarizer::SpeakerDiarizer(const SpeakerDiarizerOptions &options)
    : impl(std::make_unique<Impl>(options)) {}

SpeakerDiarizer::~SpeakerDiarizer() = default;

int32_t SpeakerDiarizer::create_stream() {
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  const int32_t stream_id = this->impl->next_stream_id++;
  Impl::StreamState state;
  state.session = std::make_unique<cppannote::StreamingDiarizationSession>(
      this->impl->engine, this->impl->session_config());
  this->impl->streams.insert({stream_id, std::move(state)});
  return stream_id;
}

void SpeakerDiarizer::free_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  this->impl->streams.erase(stream_id);
}

void SpeakerDiarizer::start_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  Impl::StreamState &state = this->impl->get_stream(stream_id);
  state.session->start_session();
  state.mapped_generation = -1;
  state.mapped_turns.clear();
  state.label_to_stable_id.clear();
  state.cached_turns.clear();
}

void SpeakerDiarizer::add_audio_to_stream(int32_t stream_id,
                                          const float *audio_data,
                                          uint64_t audio_length,
                                          int32_t sample_rate) {
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  Impl::StreamState &state = this->impl->get_stream(stream_id);
  try {
    state.session->add_audio_chunk(audio_data, (size_t)(audio_length),
                                   sample_rate);
  } catch (const std::exception &e) {
    // A clustering refresh can fail when there isn't enough speech yet (for
    // example an all-silence buffer). The audio is still cached, and the
    // refresh will be retried when more audio arrives, so this is not fatal.
    LOGF("Speaker diarization refresh failed (will retry): %s", e.what());
  }
}

std::vector<SpeakerTurn> SpeakerDiarizer::get_turns(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  Impl::StreamState &state = this->impl->get_stream(stream_id);
  this->impl->map_snapshot_to_stable_ids(state, state.session->snapshot());
  return state.cached_turns;
}

std::vector<SpeakerTurn> SpeakerDiarizer::finish_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  Impl::StreamState &state = this->impl->get_stream(stream_id);
  try {
    this->impl->map_snapshot_to_stable_ids(state, state.session->end_session());
  } catch (const std::exception &e) {
    LOGF("Final speaker diarization pass failed: %s", e.what());
  }
  return state.cached_turns;
}

std::vector<SpeakerTurn> SpeakerDiarizer::diarize(const float *audio_data,
                                                  uint64_t audio_length,
                                                  int32_t sample_rate) {
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  // A single clustering pass at the end of the audio is all that's needed,
  // so use a cadence long enough that intermediate refreshes never happen.
  constexpr double kNeverRefresh = 1e18;
  cppannote::StreamingDiarizationConfig config = this->impl->session_config();
  config.cluster_cadence = kNeverRefresh;
  Impl::StreamState state;
  state.session = std::make_unique<cppannote::StreamingDiarizationSession>(
      this->impl->engine, config);
  state.session->start_session();
  try {
    state.session->add_audio_chunk(audio_data, (size_t)(audio_length),
                                   sample_rate);
    this->impl->map_snapshot_to_stable_ids(state, state.session->end_session());
  } catch (const std::exception &e) {
    LOGF("Speaker diarization failed: %s", e.what());
  }
  return state.cached_turns;
}
