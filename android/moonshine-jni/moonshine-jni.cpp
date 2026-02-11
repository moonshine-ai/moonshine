#include <jni.h>

#include <memory>
#include <string>
#include <vector>

#include <android/log.h>
#define LOG_TAG "MoonshineJNI"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__))

#include "moonshine-c-api.h"
#include "utf8.h"

namespace {
jclass get_class(JNIEnv *env, const char *className) {
  jclass clazz = env->FindClass(className);
  if (clazz == nullptr) {
    throw std::runtime_error(std::string("Failed to find class: ") + className);
  }
  return clazz;
}

jfieldID get_field(JNIEnv *env, jclass clazz, const char *fieldName,
                   const char *fieldType) {
  jfieldID field = env->GetFieldID(clazz, fieldName, fieldType);
  if (field == nullptr) {
    throw std::runtime_error(std::string("Failed to find field: ") + fieldName);
  }
  return field;
}

jmethodID get_method(JNIEnv *env, jclass clazz, const char *methodName,
                     const char *methodSignature) {
  jmethodID method = env->GetMethodID(clazz, methodName, methodSignature);
  if (method == nullptr) {
    throw std::runtime_error(std::string("Failed to find method: ") +
                             methodName);
  }
  return method;
}

std::unique_ptr<transcript_t> c_transcript_from_jobject(
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

jobject c_transcript_to_jobject(JNIEnv *env, struct transcript_t *transcript) {
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
    std::string raw_text(line->text);
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

}  // namespace

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

    std::vector<transcriber_option_t> coptions;
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
    jbyteArray decoder_model_data, jbyteArray tokenizer_data, jint model_arch,
    jobjectArray joptions) {
  try {
    jclass optionClass = get_class(env, "ai/moonshine/voice/TranscriberOption");
    jfieldID nameField =
        get_field(env, optionClass, "name", "Ljava/lang/String;");
    jfieldID valueField =
        get_field(env, optionClass, "value", "Ljava/lang/String;");
    std::vector<transcriber_option_t> coptions;
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
    return moonshine_load_transcriber_from_memory(
        encoder_model_data_ptr, encoder_model_data_size, decoder_model_data_ptr,
        decoder_model_data_size, tokenizer_data_ptr, tokenizer_data_size,
        model_arch, coptions.data(), coptions.size(), MOONSHINE_HEADER_VERSION);
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
    jint stream_handle, jfloatArray audio_data, jint sample_rate) {
  try {
    if (audio_data == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    float *audio_data_ptr = env->GetFloatArrayElements(audio_data, nullptr);
    size_t audio_data_size = env->GetArrayLength(audio_data);
    return moonshine_transcribe_add_audio_to_stream(
        transcriber_handle, stream_handle, audio_data_ptr, audio_data_size,
        sample_rate, 0);
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
    int transcription_error = moonshine_transcribe_stream(
        transcriber_handle, stream_handle, flags, &transcript);
    if (transcription_error != 0) {
      return nullptr;
    }
    return c_transcript_to_jobject(env, transcript);
  } catch (const std::exception &e) {
    LOGE("moonshineTranscribeStream: %s\n", e.what());
    return nullptr;
  }
} 
