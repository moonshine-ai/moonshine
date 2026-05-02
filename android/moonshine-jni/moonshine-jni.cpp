#include <jni.h>

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <android/log.h>
#define LOG_TAG "MoonshineJNI"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

#include "moonshine-c-api.h"
#include "utf8.h"

static jclass get_class(JNIEnv *env, const char *className) {
  jclass clazz = env->FindClass(className);
  if (clazz == nullptr) {
    throw std::runtime_error(std::string("Failed to find class: ") + className);
  }
  return clazz;
}

static jfieldID get_field(JNIEnv *env, jclass clazz, const char *fieldName,
                          const char *fieldType) {
  jfieldID field = env->GetFieldID(clazz, fieldName, fieldType);
  if (field == nullptr) {
    throw std::runtime_error(std::string("Failed to find field: ") + fieldName);
  }
  return field;
}

static jmethodID get_method(JNIEnv *env, jclass clazz, const char *methodName,
                            const char *methodSignature) {
  jmethodID method = env->GetMethodID(clazz, methodName, methodSignature);
  if (method == nullptr) {
    throw std::runtime_error(std::string("Failed to find method: ") +
                             methodName);
  }
  return method;
}

static std::unique_ptr<transcript_t> c_transcript_from_jobject(
    JNIEnv *env, jobject javaTranscript) {
  jclass transcriptClass = env->GetObjectClass(javaTranscript);
  jfieldID linesField =
      get_field(env, transcriptClass, "lines", "Ljava/util/List;");
  jobject linesList = env->GetObjectField(javaTranscript, linesField);
  if (linesList == nullptr) {
    return nullptr;
  }

  jclass listClass = get_class(env, "java/util/List");
  jmethodID sizeMethod = get_method(env, listClass, "size", "()I");
  jmethodID getMethod =
      get_method(env, listClass, "get", "(I)Ljava/lang/Object;");

  jclass lineClass = get_class(env, "ai/moonshine/voice/TranscriptLine");

  jfieldID textField = get_field(env, lineClass, "text", "Ljava/lang/String;");
  jfieldID audioDataField = get_field(env, lineClass, "audioData", "[F");
  jfieldID startTimeField = get_field(env, lineClass, "startTime", "F");
  jfieldID durationField = get_field(env, lineClass, "duration", "F");
  jfieldID idField = get_field(env, lineClass, "id", "J");
  jfieldID isCompleteField = get_field(env, lineClass, "isComplete", "Z");
  jfieldID isUpdatedField = get_field(env, lineClass, "isUpdated", "Z");
  jfieldID isNewField = get_field(env, lineClass, "isNew", "Z");
  jfieldID hasTextChangedField =
      get_field(env, lineClass, "hasTextChanged", "Z");
  jfieldID hasSpeakerIdField = get_field(env, lineClass, "hasSpeakerId", "Z");
  jfieldID speakerIdField = get_field(env, lineClass, "speakerId", "J");
  jfieldID speakerIndexField = get_field(env, lineClass, "speakerIndex", "I");

  jsize lineCount = env->CallIntMethod(linesList, sizeMethod);
  std::unique_ptr<transcript_t> transcript(new transcript_t());
  transcript->line_count = lineCount;
  transcript->lines = new transcript_line_t[lineCount];
  for (int i = 0; i < lineCount; i++) {
    jobject line = env->CallObjectMethod(linesList, getMethod, i);
    jstring text = (jstring)env->GetObjectField(line, textField);
    transcript->lines[i].text = env->GetStringUTFChars(text, nullptr);
    jfloatArray audioData =
        (jfloatArray)env->GetObjectField(line, audioDataField);
    transcript->lines[i].audio_data =
        env->GetFloatArrayElements(audioData, nullptr);
    transcript->lines[i].audio_data_count = env->GetArrayLength(audioData);
    transcript->lines[i].start_time = env->GetFloatField(line, startTimeField);
    transcript->lines[i].duration = env->GetFloatField(line, durationField);
    transcript->lines[i].id = env->GetLongField(line, idField);
    transcript->lines[i].is_complete =
        env->GetBooleanField(line, isCompleteField);
    transcript->lines[i].is_updated =
        env->GetBooleanField(line, isUpdatedField);
    transcript->lines[i].is_new = env->GetBooleanField(line, isNewField);
    transcript->lines[i].has_text_changed =
        env->GetBooleanField(line, hasTextChangedField);
    transcript->lines[i].has_speaker_id =
        env->GetBooleanField(line, hasSpeakerIdField);
    transcript->lines[i].speaker_id = env->GetLongField(line, speakerIdField);
    transcript->lines[i].speaker_index =
        env->GetIntField(line, speakerIndexField);
  }
  return transcript;
}

static jobject c_transcript_to_jobject(JNIEnv *env, struct transcript_t *transcript) {
  jclass listClass = get_class(env, "java/util/ArrayList");
  jmethodID addMethod =
      get_method(env, listClass, "add", "(Ljava/lang/Object;)Z");

  jclass lineClass = get_class(env, "ai/moonshine/voice/TranscriptLine");
  jfieldID textField = get_field(env, lineClass, "text", "Ljava/lang/String;");
  jfieldID audioDataField = get_field(env, lineClass, "audioData", "[F");
  jfieldID startTimeField = get_field(env, lineClass, "startTime", "F");
  jfieldID durationField = get_field(env, lineClass, "duration", "F");
  jfieldID idField = get_field(env, lineClass, "id", "J");
  jfieldID isCompleteField = get_field(env, lineClass, "isComplete", "Z");
  jfieldID isUpdatedField = get_field(env, lineClass, "isUpdated", "Z");
  jfieldID isNewField = get_field(env, lineClass, "isNew", "Z");
  jfieldID hasTextChangedField =
      get_field(env, lineClass, "hasTextChanged", "Z");
  jfieldID hasSpeakerIdField = get_field(env, lineClass, "hasSpeakerId", "Z");
  jfieldID speakerIdField = get_field(env, lineClass, "speakerId", "J");
  jfieldID speakerIndexField = get_field(env, lineClass, "speakerIndex", "I");
  jmethodID listConstructor = get_method(env, listClass, "<init>", "()V");
  jobject linesList = env->NewObject(listClass, listConstructor);

  jmethodID lineConstructor = get_method(env, lineClass, "<init>", "()V");
  for (size_t i = 0; i < transcript->line_count; i++) {
    transcript_line_t *line = &transcript->lines[i];
    jobject jline = env->NewObject(lineClass, lineConstructor);
    std::string raw_text(line->text ? line->text : "");
    std::string sanitized_text = utf8::replace_invalid(raw_text);
    env->SetObjectField(jline, textField,
                        env->NewStringUTF(sanitized_text.c_str()));
    jfloatArray audioDataArray = env->NewFloatArray(line->audio_data_count);
    env->SetFloatArrayRegion(audioDataArray, 0, line->audio_data_count,
                             line->audio_data);
    env->SetObjectField(jline, audioDataField, audioDataArray);
    env->SetFloatField(jline, startTimeField, line->start_time);
    env->SetFloatField(jline, durationField, line->duration);
    env->SetLongField(jline, idField, line->id);
    env->SetBooleanField(jline, isCompleteField, line->is_complete);
    env->SetBooleanField(jline, isUpdatedField, line->is_updated);
    env->SetBooleanField(jline, isNewField, line->is_new);
    env->SetBooleanField(jline, hasTextChangedField, line->has_text_changed);
    env->SetBooleanField(jline, hasSpeakerIdField, line->has_speaker_id);
    env->SetLongField(jline, speakerIdField, line->speaker_id);
    env->SetIntField(jline, speakerIndexField, line->speaker_index);

    // Populate word timestamps if available
    if (line->words != nullptr && line->word_count > 0) {
      jclass wordClass =
          get_class(env, "ai/moonshine/voice/WordTiming");
      jmethodID wordConstructor =
          get_method(env, wordClass, "<init>", "()V");
      jfieldID wordTextField =
          get_field(env, wordClass, "word", "Ljava/lang/String;");
      jfieldID wordStartField = get_field(env, wordClass, "start", "F");
      jfieldID wordEndField = get_field(env, wordClass, "end", "F");
      jfieldID wordConfField =
          get_field(env, wordClass, "confidence", "F");

      jclass wordListClass = get_class(env, "java/util/ArrayList");
      jmethodID wordListConstructor =
          get_method(env, wordListClass, "<init>", "()V");
      jmethodID wordListAdd =
          get_method(env, wordListClass, "add", "(Ljava/lang/Object;)Z");
      jobject wordsList =
          env->NewObject(wordListClass, wordListConstructor);

      for (uint64_t j = 0; j < line->word_count; j++) {
        const transcript_word_t *w = &line->words[j];
        jobject jword = env->NewObject(wordClass, wordConstructor);
        std::string wtext(w->text ? w->text : "");
        std::string wsanitized = utf8::replace_invalid(wtext);
        env->SetObjectField(jword, wordTextField,
                            env->NewStringUTF(wsanitized.c_str()));
        env->SetFloatField(jword, wordStartField, w->start);
        env->SetFloatField(jword, wordEndField, w->end);
        env->SetFloatField(jword, wordConfField, w->confidence);
        env->CallBooleanMethod(wordsList, wordListAdd, jword);
        env->DeleteLocalRef(jword);
      }

      jfieldID wordsField = get_field(env, lineClass, "words",
                                      "Ljava/util/List;");
      env->SetObjectField(jline, wordsField, wordsList);
      env->DeleteLocalRef(wordsList);
      env->DeleteLocalRef(wordClass);
      env->DeleteLocalRef(wordListClass);
    }

    env->CallBooleanMethod(linesList, addMethod, jline);
    env->DeleteLocalRef(jline);
  }
  jclass transcriptClass = get_class(env, "ai/moonshine/voice/Transcript");
  jmethodID transcriptConstructor =
      get_method(env, transcriptClass, "<init>", "()V");
  jfieldID linesField =
      get_field(env, transcriptClass, "lines", "Ljava/util/List;");
  jobject jtranscript = env->NewObject(transcriptClass, transcriptConstructor);
  env->SetObjectField(jtranscript, linesField, linesList);

  env->DeleteLocalRef(listClass);
  env->DeleteLocalRef(lineClass);
  env->DeleteLocalRef(transcriptClass);

  return jtranscript;
}

static bool fill_moonshine_options(
    JNIEnv *env, jobjectArray joptions, std::vector<moonshine_option_t> *out_c,
    std::vector<std::pair<jstring, jstring>> *jhold) {
  if (!out_c || !jhold) {
    return false;
  }
  jhold->clear();
  out_c->clear();
  if (joptions == nullptr) {
    return true;
  }
  jclass optionClass = get_class(env, "ai/moonshine/voice/TranscriberOption");
  jfieldID nameField =
      get_field(env, optionClass, "name", "Ljava/lang/String;");
  jfieldID valueField =
      get_field(env, optionClass, "value", "Ljava/lang/String;");
  const jsize n = env->GetArrayLength(joptions);
  for (jsize i = 0; i < n; i++) {
    jobject joption = env->GetObjectArrayElement(joptions, i);
    if (joption == nullptr) {
      continue;
    }
    jstring jname = (jstring)env->GetObjectField(joption, nameField);
    jstring jvalue = (jstring)env->GetObjectField(joption, valueField);
    if (jname == nullptr || jvalue == nullptr) {
      continue;
    }
    const char *name_utf = env->GetStringUTFChars(jname, nullptr);
    const char *value_utf = env->GetStringUTFChars(jvalue, nullptr);
    jhold->push_back({jname, jvalue});
    out_c->push_back({name_utf, value_utf});
  }
  return true;
}

static void release_moonshine_options(
    JNIEnv *env, const std::vector<moonshine_option_t> &copts,
    const std::vector<std::pair<jstring, jstring>> &jhold) {
  const size_t m = std::min(copts.size(), jhold.size());
  for (size_t i = 0; i < m; i++) {
    if (jhold[i].first != nullptr) {
      env->ReleaseStringUTFChars(jhold[i].first, copts[i].name);
    }
    if (jhold[i].second != nullptr) {
      env->ReleaseStringUTFChars(jhold[i].second, copts[i].value);
    }
  }
}

static std::mutex g_tts_memory_backing_mutex;
static std::unordered_map<int32_t, std::vector<std::vector<uint8_t>>>
    g_tts_memory_backing;
static std::mutex g_g2p_memory_backing_mutex;
static std::unordered_map<int32_t, std::vector<std::vector<uint8_t>>>
    g_g2p_memory_backing;

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineGetVersion(JNIEnv * /* env */,
                                                jobject /* this */) {
  return moonshine_get_version();
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_moonshine_voice_JNI_moonshineErrorToString(JNIEnv *env,
                                                   jobject /* this */,
                                                   jint error) {
  return env->NewStringUTF(moonshine_error_to_string(error));
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_moonshine_voice_JNI_moonshineTranscriptToString(
    JNIEnv *env, jobject /* this */, jobject javaTranscript) {
  std::unique_ptr<transcript_t> transcript =
      c_transcript_from_jobject(env, javaTranscript);
  if (transcript == nullptr) {
    return env->NewStringUTF("");
  }
  jstring result =
      env->NewStringUTF(moonshine_transcript_to_string(transcript.get()));
  delete[] transcript->lines;
  return result;
}

extern "C" JNIEXPORT int JNICALL
Java_ai_moonshine_voice_JNI_moonshineLoadTranscriberFromFiles(
    JNIEnv *env, jobject /* this */, jstring path, jint model_arch,
    jobjectArray joptions) {
  try {
    jclass optionClass = get_class(env, "ai/moonshine/voice/TranscriberOption");
    jfieldID nameField =
        get_field(env, optionClass, "name", "Ljava/lang/String;");
    jfieldID valueField =
        get_field(env, optionClass, "value", "Ljava/lang/String;");

    std::vector<moonshine_option_t> coptions;
    if (joptions != nullptr) {
      for (int i = 0; i < env->GetArrayLength(joptions); i++) {
        jobject joption = env->GetObjectArrayElement(joptions, i);
        jstring jname = (jstring)env->GetObjectField(joption, nameField);
        jstring jvalue = (jstring)env->GetObjectField(joption, valueField);
        coptions.push_back({env->GetStringUTFChars(jname, nullptr),
                            env->GetStringUTFChars(jvalue, nullptr)});
      }
    }
    const char *path_str;
    if (path != nullptr) {
      path_str = env->GetStringUTFChars(path, nullptr);
    } else {
      path_str = nullptr;
    }
    return moonshine_load_transcriber_from_files(
        path_str, model_arch, coptions.data(), coptions.size(),
        MOONSHINE_HEADER_VERSION);
  } catch (const std::exception &e) {
    LOGE("moonshineLoadTranscriberFromFiles: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT int JNICALL
Java_ai_moonshine_voice_JNI_moonshineLoadTranscriberFromMemory(
    JNIEnv *env, jobject /* this */, jbyteArray encoder_model_data,
    jbyteArray decoder_model_data, jbyteArray tokenizer_data,
    jbyteArray spelling_model_data, jint model_arch, jobjectArray joptions) {
  try {
    jclass optionClass = get_class(env, "ai/moonshine/voice/TranscriberOption");
    jfieldID nameField =
        get_field(env, optionClass, "name", "Ljava/lang/String;");
    jfieldID valueField =
        get_field(env, optionClass, "value", "Ljava/lang/String;");
    std::vector<moonshine_option_t> coptions;
    if (joptions != nullptr) {
      for (int i = 0; i < env->GetArrayLength(joptions); i++) {
        jobject joption = env->GetObjectArrayElement(joptions, i);
        jstring jname = (jstring)env->GetObjectField(joption, nameField);
        jstring jvalue = (jstring)env->GetObjectField(joption, valueField);
        coptions.push_back({env->GetStringUTFChars(jname, nullptr),
                            env->GetStringUTFChars(jvalue, nullptr)});
      }
    }
    const uint8_t *encoder_model_data_ptr =
        (uint8_t *)(env->GetByteArrayElements(encoder_model_data, nullptr));
    size_t encoder_model_data_size = env->GetArrayLength(encoder_model_data);
    const uint8_t *decoder_model_data_ptr =
        (uint8_t *)(env->GetByteArrayElements(decoder_model_data, nullptr));
    size_t decoder_model_data_size = env->GetArrayLength(decoder_model_data);
    const uint8_t *tokenizer_data_ptr =
        (uint8_t *)(env->GetByteArrayElements(tokenizer_data, nullptr));
    size_t tokenizer_data_size = env->GetArrayLength(tokenizer_data);
    const uint8_t *spelling_model_data_ptr = nullptr;
    size_t spelling_model_data_size = 0;
    if (spelling_model_data != nullptr) {
      spelling_model_data_ptr =
          (uint8_t *)(env->GetByteArrayElements(spelling_model_data, nullptr));
      spelling_model_data_size = env->GetArrayLength(spelling_model_data);
    }
    return moonshine_load_transcriber_from_memory(
        encoder_model_data_ptr, encoder_model_data_size, decoder_model_data_ptr,
        decoder_model_data_size, tokenizer_data_ptr, tokenizer_data_size,
        spelling_model_data_ptr, spelling_model_data_size, model_arch,
        coptions.data(), coptions.size(), MOONSHINE_HEADER_VERSION);
  } catch (const std::exception &e) {
    LOGE("moonshineLoadTranscriberFromMemory: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_moonshine_voice_JNI_moonshineFreeTranscriber(JNIEnv * /* env */,
                                                     jobject /* this */,
                                                     jint transcriber_handle) {
  try {
    moonshine_free_transcriber(transcriber_handle);
  } catch (const std::exception &e) {
    LOGE("moonshineFreeTranscriber: %s\n", e.what());
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_ai_moonshine_voice_JNI_moonshineTranscribeWithoutStreaming(
    JNIEnv *env, jobject /* this */, jint transcriber_handle,
    jfloatArray audio_data, jint sample_rate, jint flags) {
  try {
    float *audio_data_ptr = env->GetFloatArrayElements(audio_data, nullptr);
    size_t audio_data_size = env->GetArrayLength(audio_data);
    struct transcript_t *transcript = nullptr;
    int transcription_error = moonshine_transcribe_without_streaming(
        transcriber_handle, audio_data_ptr, audio_data_size, sample_rate, flags,
        &transcript);
    if (transcription_error != 0) {
      return nullptr;
    }
    return c_transcript_to_jobject(env, transcript);
  } catch (const std::exception &e) {
    LOGE("moonshineTranscribeWithoutStreaming: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT int JNICALL
Java_ai_moonshine_voice_JNI_moonshineCreateStream(JNIEnv * /* env */,
                                                  jobject /* this */,
                                                  jint transcriber_handle) {
  try {
    return moonshine_create_stream(transcriber_handle, 0);
  } catch (const std::exception &e) {
    LOGE("moonshineCreateStream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_moonshine_voice_JNI_moonshineFreeStream(JNIEnv * /* env */,
                                                jobject /* this */,
                                                jint transcriber_handle,
                                                jint stream_handle) {
  try {
    moonshine_free_stream(transcriber_handle, stream_handle);
  } catch (const std::exception &e) {
    LOGE("moonshineFreeStream: %s\n", e.what());
  }
}

extern "C" JNIEXPORT int JNICALL
Java_ai_moonshine_voice_JNI_moonshineStartStream(JNIEnv * /* env */,
                                                 jobject /* this */,
                                                 jint transcriber_handle,
                                                 jint stream_handle) {
  try {
    return moonshine_start_stream(transcriber_handle, stream_handle);
  } catch (const std::exception &e) {
    LOGE("moonshineStartStream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT int JNICALL
Java_ai_moonshine_voice_JNI_moonshineStopStream(JNIEnv * /* env */,
                                                jobject /* this */,
                                                jint transcriber_handle,
                                                jint stream_handle) {
  try {
    return moonshine_stop_stream(transcriber_handle, stream_handle);
  } catch (const std::exception &e) {
    LOGE("moonshineStopStream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT int JNICALL
Java_ai_moonshine_voice_JNI_moonshineAddAudioToStream(
    JNIEnv *env, jobject /* this */, jint transcriber_handle,
    jint stream_handle, jfloatArray audio_data, jint sample_rate, jint flags) {
  try {
    if (audio_data == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    float *audio_data_ptr = env->GetFloatArrayElements(audio_data, nullptr);
    size_t audio_data_size = env->GetArrayLength(audio_data);
    return moonshine_transcribe_add_audio_to_stream(
        transcriber_handle, stream_handle, audio_data_ptr, audio_data_size,
        sample_rate, flags);
  } catch (const std::exception &e) {
    LOGE("moonshineAddAudioToStream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_ai_moonshine_voice_JNI_moonshineTranscribeStream(JNIEnv *env,
                                                      jobject /* this */,
                                                      jint transcriber_handle,
                                                      jint stream_handle,
                                                      jint flags) {
  try {
    struct transcript_t *transcript = nullptr;
    LOGE("moonshineTranscribeStream: start transcribe stream");
    int transcription_error = moonshine_transcribe_stream(
        transcriber_handle, stream_handle, flags, &transcript);
    LOGE("moonshineTranscribeStream: transcription error: %d", transcription_error);
    if (transcription_error != 0) {
      LOGE("moonshineTranscribeStream: transcription error: %d", transcription_error);
      return nullptr;
    }
    LOGE("moonshineTranscribeStream: transcript=%p", (void *)transcript);
    return c_transcript_to_jobject(env, transcript);
  } catch (const std::exception &e) {
    LOGE("moonshineTranscribeStream: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineCreateTtsSynthesizerFromFiles(
    JNIEnv *env, jobject /* this */, jstring language, jobjectArray jfilenames,
    jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const char *lang_ptr = nullptr;
    if (language != nullptr) {
      lang_ptr = env->GetStringUTFChars(language, nullptr);
    }
    const char **c_filenames_ptr = nullptr;
    uint64_t filenames_count = 0;
    std::vector<std::string> filename_storage;
    std::vector<const char *> c_filenames;
    if (jfilenames != nullptr) {
      const jsize n = env->GetArrayLength(jfilenames);
      filename_storage.reserve(static_cast<size_t>(n));
      c_filenames.reserve(static_cast<size_t>(n));
      for (jsize i = 0; i < n; i++) {
        jstring jf = (jstring)env->GetObjectArrayElement(jfilenames, i);
        if (jf == nullptr) {
          release_moonshine_options(env, copts, jhold);
          if (lang_ptr != nullptr) {
            env->ReleaseStringUTFChars(language, lang_ptr);
          }
          return MOONSHINE_ERROR_INVALID_ARGUMENT;
        }
        const char *u = env->GetStringUTFChars(jf, nullptr);
        filename_storage.emplace_back(u);
        env->ReleaseStringUTFChars(jf, u);
        c_filenames.push_back(filename_storage.back().c_str());
      }
      filenames_count = static_cast<uint64_t>(c_filenames.size());
      c_filenames_ptr = c_filenames.data();
    }
    const int32_t handle = moonshine_create_tts_synthesizer_from_files(
        lang_ptr, c_filenames_ptr, filenames_count, copts.data(), copts.size(),
        MOONSHINE_HEADER_VERSION);
    release_moonshine_options(env, copts, jhold);
    if (lang_ptr != nullptr) {
      env->ReleaseStringUTFChars(language, lang_ptr);
    }
    return handle;
  } catch (const std::exception &e) {
    LOGE("moonshineCreateTtsSynthesizerFromFiles: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineCreateTtsSynthesizerFromMemory(
    JNIEnv *env, jobject /* this */, jstring language, jobjectArray jfilenames,
    jobjectArray jmemory, jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const char *lang_ptr = nullptr;
    if (language != nullptr) {
      lang_ptr = env->GetStringUTFChars(language, nullptr);
    }
    const jsize n = jfilenames != nullptr ? env->GetArrayLength(jfilenames) : 0;
    if (n > 0 && jmemory == nullptr) {
      release_moonshine_options(env, copts, jhold);
      if (lang_ptr != nullptr) {
        env->ReleaseStringUTFChars(language, lang_ptr);
      }
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    if (n > 0 && env->GetArrayLength(jmemory) != n) {
      release_moonshine_options(env, copts, jhold);
      if (lang_ptr != nullptr) {
        env->ReleaseStringUTFChars(language, lang_ptr);
      }
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    std::vector<std::string> filename_storage;
    std::vector<const char *> c_filenames;
    filename_storage.reserve(static_cast<size_t>(n));
    c_filenames.reserve(static_cast<size_t>(n));
    std::vector<std::vector<uint8_t>> backing;
    std::vector<const uint8_t *> c_mem;
    std::vector<uint64_t> c_sizes;
    c_mem.reserve(static_cast<size_t>(n));
    c_sizes.reserve(static_cast<size_t>(n));
    for (jsize i = 0; i < n; i++) {
      jstring jf = (jstring)env->GetObjectArrayElement(jfilenames, i);
      if (jf == nullptr) {
        release_moonshine_options(env, copts, jhold);
        if (lang_ptr != nullptr) {
          env->ReleaseStringUTFChars(language, lang_ptr);
        }
        return MOONSHINE_ERROR_INVALID_ARGUMENT;
      }
      const char *u = env->GetStringUTFChars(jf, nullptr);
      filename_storage.emplace_back(u);
      env->ReleaseStringUTFChars(jf, u);
      c_filenames.push_back(filename_storage.back().c_str());
      jbyteArray jbuf = (jbyteArray)env->GetObjectArrayElement(jmemory, i);
      if (jbuf == nullptr) {
        c_mem.push_back(nullptr);
        c_sizes.push_back(0);
        continue;
      }
      const jsize len = env->GetArrayLength(jbuf);
      if (len <= 0) {
        c_mem.push_back(nullptr);
        c_sizes.push_back(0);
        continue;
      }
      std::vector<uint8_t> copy(static_cast<size_t>(len));
      env->GetByteArrayRegion(jbuf, 0, len,
                              reinterpret_cast<jbyte *>(copy.data()));
      backing.push_back(std::move(copy));
      c_mem.push_back(backing.back().data());
      c_sizes.push_back(static_cast<uint64_t>(backing.back().size()));
    }
    const int32_t handle = moonshine_create_tts_synthesizer_from_memory(
        lang_ptr, c_filenames.data(), static_cast<uint64_t>(c_filenames.size()),
        c_mem.data(), c_sizes.data(), copts.data(), copts.size(),
        MOONSHINE_HEADER_VERSION);
    release_moonshine_options(env, copts, jhold);
    if (lang_ptr != nullptr) {
      env->ReleaseStringUTFChars(language, lang_ptr);
    }
    if (handle >= 0 && !backing.empty()) {
      std::lock_guard<std::mutex> lock(g_tts_memory_backing_mutex);
      g_tts_memory_backing[handle] = std::move(backing);
    }
    return handle;
  } catch (const std::exception &e) {
    LOGE("moonshineCreateTtsSynthesizerFromMemory: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_moonshine_voice_JNI_moonshineFreeTtsSynthesizer(JNIEnv * /* env */,
                                                        jobject /* this */,
                                                        jint tts_handle) {
  try {
    moonshine_free_tts_synthesizer(tts_handle);
    std::lock_guard<std::mutex> lock(g_tts_memory_backing_mutex);
    g_tts_memory_backing.erase(tts_handle);
  } catch (const std::exception &e) {
    LOGE("moonshineFreeTtsSynthesizer: %s\n", e.what());
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_moonshine_voice_JNI_moonshineGetG2pDependencies(JNIEnv *env,
                                                      jobject /* this */,
                                                      jstring languages,
                                                      jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return nullptr;
    }
    const char *lang_ptr = nullptr;
    if (languages != nullptr) {
      lang_ptr = env->GetStringUTFChars(languages, nullptr);
    }
    char *out = nullptr;
    const int32_t err = moonshine_get_g2p_dependencies(
        lang_ptr, copts.data(), copts.size(), &out);
    release_moonshine_options(env, copts, jhold);
    if (languages != nullptr && lang_ptr != nullptr) {
      env->ReleaseStringUTFChars(languages, lang_ptr);
    }
    if (err != MOONSHINE_ERROR_NONE) {
      if (out != nullptr) {
        std::free(out);
      }
      return nullptr;
    }
    if (out == nullptr) {
      return env->NewStringUTF("");
    }
    std::string sanitized = utf8::replace_invalid(std::string(out));
    std::free(out);
    return env->NewStringUTF(sanitized.c_str());
  } catch (const std::exception &e) {
    LOGE("moonshineGetG2pDependencies: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_moonshine_voice_JNI_moonshineGetTtsDependencies(JNIEnv *env,
                                                      jobject /* this */,
                                                      jstring languages,
                                                      jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return nullptr;
    }
    const char *lang_ptr = nullptr;
    if (languages != nullptr) {
      lang_ptr = env->GetStringUTFChars(languages, nullptr);
    }
    char *out = nullptr;
    const int32_t err = moonshine_get_tts_dependencies(
        lang_ptr, copts.data(), copts.size(), &out);
    release_moonshine_options(env, copts, jhold);
    if (languages != nullptr && lang_ptr != nullptr) {
      env->ReleaseStringUTFChars(languages, lang_ptr);
    }
    if (err != MOONSHINE_ERROR_NONE) {
      if (out != nullptr) {
        std::free(out);
      }
      return nullptr;
    }
    if (out == nullptr) {
      return env->NewStringUTF("");
    }
    std::string sanitized = utf8::replace_invalid(std::string(out));
    std::free(out);
    return env->NewStringUTF(sanitized.c_str());
  } catch (const std::exception &e) {
    LOGE("moonshineGetTtsDependencies: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_moonshine_voice_JNI_moonshineGetTtsVoices(JNIEnv *env, jobject /* this */,
                                                  jstring languages,
                                                  jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return nullptr;
    }
    const char *lang_ptr = nullptr;
    if (languages != nullptr) {
      lang_ptr = env->GetStringUTFChars(languages, nullptr);
    }
    char *out = nullptr;
    const int32_t err = moonshine_get_tts_voices(lang_ptr, copts.data(),
                                                 copts.size(), &out);
    release_moonshine_options(env, copts, jhold);
    if (languages != nullptr && lang_ptr != nullptr) {
      env->ReleaseStringUTFChars(languages, lang_ptr);
    }
    if (err != MOONSHINE_ERROR_NONE) {
      if (out != nullptr) {
        std::free(out);
      }
      return nullptr;
    }
    if (out == nullptr) {
      return env->NewStringUTF("{}");
    }
    std::string sanitized = utf8::replace_invalid(std::string(out));
    std::free(out);
    return env->NewStringUTF(sanitized.c_str());
  } catch (const std::exception &e) {
    LOGE("moonshineGetTtsVoices: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_ai_moonshine_voice_JNI_moonshineTextToSpeech(JNIEnv *env, jobject /* this */,
                                                  jint tts_handle, jstring text,
                                                  jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return nullptr;
    }
    if (text == nullptr) {
      release_moonshine_options(env, copts, jhold);
      return nullptr;
    }
    const char *text_ptr = env->GetStringUTFChars(text, nullptr);
    float *out_audio = nullptr;
    uint64_t out_size = 0;
    int32_t out_sr = 0;
    const int32_t err = moonshine_text_to_speech(
        tts_handle, text_ptr, copts.data(), copts.size(), &out_audio, &out_size,
        &out_sr);
    env->ReleaseStringUTFChars(text, text_ptr);
    release_moonshine_options(env, copts, jhold);
    if (err != MOONSHINE_ERROR_NONE) {
      if (out_audio != nullptr) {
        std::free(out_audio);
      }
      return nullptr;
    }
    jclass resClass = get_class(env, "ai/moonshine/voice/TtsSynthesisResult");
    jmethodID ctor = get_method(env, resClass, "<init>", "()V");
    jobject res = env->NewObject(resClass, ctor);
    jfieldID samplesField = get_field(env, resClass, "samples", "[F");
    jfieldID srField = get_field(env, resClass, "sampleRateHz", "I");
    jfloatArray jsamples = env->NewFloatArray(static_cast<jsize>(out_size));
    if (out_audio != nullptr && out_size > 0) {
      env->SetFloatArrayRegion(jsamples, 0, static_cast<jsize>(out_size),
                               out_audio);
      std::free(out_audio);
    }
    env->SetObjectField(res, samplesField, jsamples);
    env->SetIntField(res, srField, out_sr);
    env->DeleteLocalRef(resClass);
    return res;
  } catch (const std::exception &e) {
    LOGE("moonshineTextToSpeech: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineCreateGraphemeToPhonemizerFromFiles(
    JNIEnv *env, jobject /* this */, jstring language, jobjectArray jfilenames,
    jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const char *lang_ptr = nullptr;
    if (language != nullptr) {
      lang_ptr = env->GetStringUTFChars(language, nullptr);
    }
    const char **c_filenames_ptr = nullptr;
    uint64_t filenames_count = 0;
    std::vector<std::string> filename_storage;
    std::vector<const char *> c_filenames;
    if (jfilenames != nullptr) {
      const jsize n = env->GetArrayLength(jfilenames);
      filename_storage.reserve(static_cast<size_t>(n));
      c_filenames.reserve(static_cast<size_t>(n));
      for (jsize i = 0; i < n; i++) {
        jstring jf = (jstring)env->GetObjectArrayElement(jfilenames, i);
        if (jf == nullptr) {
          release_moonshine_options(env, copts, jhold);
          if (lang_ptr != nullptr) {
            env->ReleaseStringUTFChars(language, lang_ptr);
          }
          return MOONSHINE_ERROR_INVALID_ARGUMENT;
        }
        const char *u = env->GetStringUTFChars(jf, nullptr);
        filename_storage.emplace_back(u);
        env->ReleaseStringUTFChars(jf, u);
        c_filenames.push_back(filename_storage.back().c_str());
      }
      filenames_count = static_cast<uint64_t>(c_filenames.size());
      c_filenames_ptr = c_filenames.data();
    }
    const int32_t handle = moonshine_create_grapheme_to_phonemizer_from_files(
        lang_ptr, c_filenames_ptr, filenames_count, copts.data(), copts.size(),
        MOONSHINE_HEADER_VERSION);
    release_moonshine_options(env, copts, jhold);
    if (lang_ptr != nullptr) {
      env->ReleaseStringUTFChars(language, lang_ptr);
    }
    return handle;
  } catch (const std::exception &e) {
    LOGE("moonshineCreateGraphemeToPhonemizerFromFiles: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineCreateGraphemeToPhonemizerFromMemory(
    JNIEnv *env, jobject /* this */, jstring language, jobjectArray jfilenames,
    jobjectArray jmemory, jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const char *lang_ptr = nullptr;
    if (language != nullptr) {
      lang_ptr = env->GetStringUTFChars(language, nullptr);
    }
    const jsize n = jfilenames != nullptr ? env->GetArrayLength(jfilenames) : 0;
    if (n > 0 && jmemory == nullptr) {
      release_moonshine_options(env, copts, jhold);
      if (lang_ptr != nullptr) {
        env->ReleaseStringUTFChars(language, lang_ptr);
      }
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    if (n > 0 && env->GetArrayLength(jmemory) != n) {
      release_moonshine_options(env, copts, jhold);
      if (lang_ptr != nullptr) {
        env->ReleaseStringUTFChars(language, lang_ptr);
      }
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    std::vector<std::string> filename_storage;
    std::vector<const char *> c_filenames;
    filename_storage.reserve(static_cast<size_t>(n));
    c_filenames.reserve(static_cast<size_t>(n));
    std::vector<std::vector<uint8_t>> backing;
    std::vector<const uint8_t *> c_mem;
    std::vector<uint64_t> c_sizes;
    c_mem.reserve(static_cast<size_t>(n));
    c_sizes.reserve(static_cast<size_t>(n));
    for (jsize i = 0; i < n; i++) {
      jstring jf = (jstring)env->GetObjectArrayElement(jfilenames, i);
      if (jf == nullptr) {
        release_moonshine_options(env, copts, jhold);
        if (lang_ptr != nullptr) {
          env->ReleaseStringUTFChars(language, lang_ptr);
        }
        return MOONSHINE_ERROR_INVALID_ARGUMENT;
      }
      const char *u = env->GetStringUTFChars(jf, nullptr);
      filename_storage.emplace_back(u);
      env->ReleaseStringUTFChars(jf, u);
      c_filenames.push_back(filename_storage.back().c_str());
      jbyteArray jbuf = (jbyteArray)env->GetObjectArrayElement(jmemory, i);
      if (jbuf == nullptr) {
        c_mem.push_back(nullptr);
        c_sizes.push_back(0);
        continue;
      }
      const jsize len = env->GetArrayLength(jbuf);
      if (len <= 0) {
        c_mem.push_back(nullptr);
        c_sizes.push_back(0);
        continue;
      }
      std::vector<uint8_t> copy(static_cast<size_t>(len));
      env->GetByteArrayRegion(jbuf, 0, len,
                              reinterpret_cast<jbyte *>(copy.data()));
      backing.push_back(std::move(copy));
      c_mem.push_back(backing.back().data());
      c_sizes.push_back(static_cast<uint64_t>(backing.back().size()));
    }
    const int32_t handle = moonshine_create_grapheme_to_phonemizer_from_memory(
        lang_ptr, c_filenames.data(), static_cast<uint64_t>(c_filenames.size()),
        c_mem.data(), c_sizes.data(), copts.data(), copts.size(),
        MOONSHINE_HEADER_VERSION);
    release_moonshine_options(env, copts, jhold);
    if (lang_ptr != nullptr) {
      env->ReleaseStringUTFChars(language, lang_ptr);
    }
    if (handle >= 0 && !backing.empty()) {
      std::lock_guard<std::mutex> lock(g_g2p_memory_backing_mutex);
      g_g2p_memory_backing[handle] = std::move(backing);
    }
    return handle;
  } catch (const std::exception &e) {
    LOGE("moonshineCreateGraphemeToPhonemizerFromMemory: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_moonshine_voice_JNI_moonshineFreeGraphemeToPhonemizer(
    JNIEnv * /* env */, jobject /* this */, jint g2p_handle) {
  try {
    moonshine_free_grapheme_to_phonemizer(g2p_handle);
    std::lock_guard<std::mutex> lock(g_g2p_memory_backing_mutex);
    g_g2p_memory_backing.erase(g2p_handle);
  } catch (const std::exception &e) {
    LOGE("moonshineFreeGraphemeToPhonemizer: %s\n", e.what());
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_moonshine_voice_JNI_moonshineTextToPhonemes(
    JNIEnv *env, jobject /* this */, jint g2p_handle, jstring text,
    jobjectArray joptions) {
  try {
    std::vector<moonshine_option_t> copts;
    std::vector<std::pair<jstring, jstring>> jhold;
    if (!fill_moonshine_options(env, joptions, &copts, &jhold)) {
      return nullptr;
    }
    if (text == nullptr) {
      release_moonshine_options(env, copts, jhold);
      return nullptr;
    }
    const char *text_ptr = env->GetStringUTFChars(text, nullptr);
    const char *out_ph = nullptr;
    uint64_t out_count = 0;
    const int32_t err = moonshine_text_to_phonemes(
        g2p_handle, text_ptr, copts.data(), copts.size(), &out_ph, &out_count);
    env->ReleaseStringUTFChars(text, text_ptr);
    release_moonshine_options(env, copts, jhold);
    if (err != MOONSHINE_ERROR_NONE) {
      if (out_ph != nullptr) {
        std::free(const_cast<char *>(out_ph));
      }
      return nullptr;
    }
    if (out_ph == nullptr || out_count == 0) {
      return env->NewStringUTF("");
    }
    std::string ipa(out_ph);
    std::free(const_cast<char *>(out_ph));
    std::string sanitized = utf8::replace_invalid(ipa);
    return env->NewStringUTF(sanitized.c_str());
  } catch (const std::exception &e) {
    LOGE("moonshineTextToPhonemes: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineCreateIntentRecognizer(
    JNIEnv *env, jobject /* this */, jstring model_path,
    jint embedding_arch, jstring model_variant) {
  try {
    if (model_path == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const char *path_ptr = env->GetStringUTFChars(model_path, nullptr);
    const char *var_ptr = nullptr;
    if (model_variant != nullptr) {
      var_ptr = env->GetStringUTFChars(model_variant, nullptr);
    }
    const int32_t handle = moonshine_create_intent_recognizer(
        path_ptr, static_cast<uint32_t>(embedding_arch),
        var_ptr != nullptr ? var_ptr : "q4");
    env->ReleaseStringUTFChars(model_path, path_ptr);
    if (model_variant != nullptr && var_ptr != nullptr) {
      env->ReleaseStringUTFChars(model_variant, var_ptr);
    }
    return handle;
  } catch (const std::exception &e) {
    LOGE("moonshineCreateIntentRecognizer: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_moonshine_voice_JNI_moonshineFreeIntentRecognizer(
    JNIEnv * /* env */, jobject /* this */, jint intent_handle) {
  moonshine_free_intent_recognizer(intent_handle);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineRegisterIntent(JNIEnv *env, jobject /* this */,
                                                    jint intent_handle,
                                                    jstring canonical_phrase,
                                                    jfloatArray embedding,
                                                    jint priority) {
  try {
    if (canonical_phrase == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const char *phrase = env->GetStringUTFChars(canonical_phrase, nullptr);
    float *emb_ptr = nullptr;
    uint64_t emb_size = 0;
    if (embedding != nullptr) {
      emb_ptr = env->GetFloatArrayElements(embedding, nullptr);
      emb_size = static_cast<uint64_t>(env->GetArrayLength(embedding));
    }
    const int32_t err = moonshine_register_intent(intent_handle, phrase,
                                                   emb_ptr, emb_size, priority);
    if (emb_ptr != nullptr) {
      env->ReleaseFloatArrayElements(embedding, emb_ptr, JNI_ABORT);
    }
    env->ReleaseStringUTFChars(canonical_phrase, phrase);
    return err;
  } catch (const std::exception &e) {
    LOGE("moonshineRegisterIntent: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineUnregisterIntent(
    JNIEnv *env, jobject /* this */, jint intent_handle,
    jstring canonical_phrase) {
  try {
    if (canonical_phrase == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const char *phrase = env->GetStringUTFChars(canonical_phrase, nullptr);
    const int32_t err = moonshine_unregister_intent(intent_handle, phrase);
    env->ReleaseStringUTFChars(canonical_phrase, phrase);
    return err;
  } catch (const std::exception &e) {
    LOGE("moonshineUnregisterIntent: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_moonshine_voice_JNI_moonshineGetClosestIntents(
    JNIEnv *env, jobject /* this */, jint intent_handle, jstring utterance,
    jfloat tolerance) {
  try {
    if (utterance == nullptr) {
      return nullptr;
    }
    const char *utter = env->GetStringUTFChars(utterance, nullptr);
    moonshine_intent_match_t *matches = nullptr;
    uint64_t count = 0;
    const int32_t err = moonshine_get_closest_intents(
        intent_handle, utter, tolerance, &matches, &count);
    env->ReleaseStringUTFChars(utterance, utter);
    if (err != MOONSHINE_ERROR_NONE) {
      if (matches != nullptr) {
        moonshine_free_intent_matches(matches, count);
      }
      return nullptr;
    }

    jclass match_class = get_class(env, "ai/moonshine/voice/IntentMatch");
    jmethodID ctor =
        get_method(env, match_class, "<init>", "(Ljava/lang/String;F)V");
    const jsize n = static_cast<jsize>(count);
    jobjectArray arr = env->NewObjectArray(n, match_class, nullptr);
    if (arr == nullptr) {
      moonshine_free_intent_matches(matches, count);
      env->DeleteLocalRef(match_class);
      return nullptr;
    }
    for (jsize i = 0; i < n; ++i) {
      const char *ph =
          matches[i].canonical_phrase ? matches[i].canonical_phrase : "";
      jstring jphrase = env->NewStringUTF(ph);
      jobject obj =
          env->NewObject(match_class, ctor, jphrase, matches[i].similarity);
      env->DeleteLocalRef(jphrase);
      env->SetObjectArrayElement(arr, i, obj);
      env->DeleteLocalRef(obj);
    }
    moonshine_free_intent_matches(matches, count);
    env->DeleteLocalRef(match_class);
    return arr;
  } catch (const std::exception &e) {
    LOGE("moonshineGetClosestIntents: %s\n", e.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineGetIntentCount(JNIEnv * /* env */,
                                                    jobject /* this */,
                                                    jint intent_handle) {
  return moonshine_get_intent_count(intent_handle);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_moonshine_voice_JNI_moonshineClearIntents(JNIEnv * /* env */,
                                                jobject /* this */,
                                                jint intent_handle) {
  try {
    return moonshine_clear_intents(intent_handle);
  } catch (const std::exception &e) {
    LOGE("moonshineClearIntents: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_ai_moonshine_voice_JNI_moonshineCalculateIntentEmbedding(
    JNIEnv *env, jobject /* this */, jint intent_handle, jstring sentence) {
  try {
    if (sentence == nullptr) {
      return nullptr;
    }
    const char *sent = env->GetStringUTFChars(sentence, nullptr);
    float *out_embedding = nullptr;
    uint64_t out_size = 0;
    const int32_t err = moonshine_calculate_intent_embedding(
        intent_handle, sent, &out_embedding, &out_size, nullptr);
    env->ReleaseStringUTFChars(sentence, sent);
    if (err != MOONSHINE_ERROR_NONE || out_embedding == nullptr) {
      moonshine_free_intent_embedding(out_embedding);
      return nullptr;
    }
    const jsize n = static_cast<jsize>(out_size);
    jfloatArray result = env->NewFloatArray(n);
    if (result != nullptr) {
      env->SetFloatArrayRegion(result, 0, n, out_embedding);
    }
    moonshine_free_intent_embedding(out_embedding);
    return result;
  } catch (const std::exception &e) {
    LOGE("moonshineCalculateIntentEmbedding: %s\n", e.what());
    return nullptr;
  }
}

