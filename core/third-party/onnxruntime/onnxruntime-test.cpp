#include <onnxruntime_c_api.h>

#include <cstdio>
#include <string>
#include <vector>

int main(int, char **) {
  const char *version = OrtGetApiBase()->GetVersionString();
  printf("Onnxruntime version: %s\n", version);

  const OrtApi *ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (ort_api == nullptr) {
    fprintf(stderr, "Failed to get OrtApi\n");
    return 1;
  }

  OrtEnv *ort_env = nullptr;
  OrtStatus *env_status = ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                             "OnnxruntimeTest", &ort_env);
  if (env_status != NULL) {
    const char *msg = ort_api->GetErrorMessage(env_status);
    fprintf(stderr, "ORT Error: %s at %s:%d\n", msg, __FILE__, __LINE__);
    ort_api->ReleaseStatus(env_status);
    return -1;
  }
  ort_api->ReleaseEnv(ort_env);

  OrtSessionOptions *session_options = nullptr;
  OrtStatus *session_options_status =
      ort_api->CreateSessionOptions(&session_options);
  if (session_options_status != NULL) {
    const char *msg = ort_api->GetErrorMessage(session_options_status);
    fprintf(stderr, "ORT Error: %s at %s:%d\n", msg, __FILE__, __LINE__);
    ort_api->ReleaseStatus(session_options_status);
    return -1;
  }
  ort_api->ReleaseSessionOptions(session_options);

  return 0;
}