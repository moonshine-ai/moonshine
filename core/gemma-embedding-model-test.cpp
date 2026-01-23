#include "gemma-embedding-model.h"

#include <cmath>
#include <filesystem>
#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("gemma-embedding-model") {
  // Check if test model exists
  std::string model_dir = "embeddinggemma-300m-ONNX";
  if (!std::filesystem::exists(model_dir)) {
    MESSAGE("Skipping Gemma embedding tests - model not found at: ", model_dir);
    return;
  }

  SUBCASE("load model") {
    GemmaEmbeddingModel model;
    // Use q4 quantized model for faster loading
    int result = model.load(model_dir.c_str(), "q4");
    CHECK(result == 0);
    CHECK(model.is_loaded() == true);
  }

  SUBCASE("get embeddings") {
    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "q4");
    REQUIRE(result == 0);

    std::vector<float> embedding = model.get_embeddings("Hello, world!");

    // Check that we got an embedding
    CHECK(embedding.size() > 0);
    MESSAGE("Embedding dimension: ", embedding.size());

    // Check that embedding is normalized (length ~= 1)
    float norm = 0.0f;
    for (float v : embedding) {
      norm += v * v;
    }
    norm = std::sqrt(norm);
    CHECK(norm == doctest::Approx(1.0f).epsilon(0.01f));
  }

  SUBCASE("identical strings have similarity 1.0") {
    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "q4");
    REQUIRE(result == 0);

    float similarity = model.get_similarity("Hello world", "Hello world");
    CHECK(similarity == doctest::Approx(1.0f).epsilon(0.001f));
  }

  SUBCASE("similar strings have high similarity") {
    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "q4");
    REQUIRE(result == 0);

    float similarity = model.get_similarity("Mars is known as the Red Planet",
                                            "The Red Planet is Mars");

    MESSAGE("Similarity between similar strings: ", similarity);
    CHECK(similarity > 0.7f);
  }

  SUBCASE("different strings have lower similarity") {
    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "q4");
    REQUIRE(result == 0);

    float similarity = model.get_similarity("Mars is known as the Red Planet",
                                            "I love eating pizza");

    MESSAGE("Similarity between different strings: ", similarity);
    CHECK(similarity < 0.5f);
  }

  SUBCASE("query and document embeddings") {
    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "q4");
    REQUIRE(result == 0);

    std::vector<float> query_emb =
        model.get_query_embeddings("Which planet is known as the Red Planet?");
    std::vector<float> doc_emb = model.get_document_embeddings(
        "Mars, known for its reddish appearance, is often referred to as the "
        "Red Planet.");

    CHECK(query_emb.size() > 0);
    CHECK(doc_emb.size() > 0);
    CHECK(query_emb.size() == doc_emb.size());

    // Compute similarity
    float similarity = model.get_similarity(query_emb, doc_emb);
    MESSAGE("Query-document similarity: ", similarity);
  }

  SUBCASE("truncate embedding with MRL") {
    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "q4");
    REQUIRE(result == 0);

    std::vector<float> full_embedding = model.get_embeddings("Test text");
    REQUIRE(full_embedding.size() > 0);

    // Truncate to different sizes
    for (int target_dim : {128, 256, 512}) {
      if (static_cast<size_t>(target_dim) >= full_embedding.size()) continue;

      std::vector<float> truncated =
          GemmaEmbeddingModel::truncate_embedding(full_embedding, target_dim);

      CHECK(truncated.size() == static_cast<size_t>(target_dim));

      // Check normalization
      float norm = 0.0f;
      for (float v : truncated) {
        norm += v * v;
      }
      norm = std::sqrt(norm);
      CHECK(norm == doctest::Approx(1.0f).epsilon(0.01f));
    }
  }

  SUBCASE("config values") {
    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "q4");
    REQUIRE(result == 0);

    const GemmaEmbeddingConfig &config = model.get_config();
    CHECK(config.embedding_dim == 768);
    CHECK(config.max_seq_length == 2048);
    CHECK(config.vocab_size == 262144);
  }
}

TEST_CASE("gemma-embedding-model error handling") {
  SUBCASE("load nonexistent model") {
    GemmaEmbeddingModel model;
    int result = model.load("/nonexistent/path", "q4");
    CHECK(result != 0);
    CHECK(model.is_loaded() == false);
  }

  SUBCASE("get embeddings without loading") {
    GemmaEmbeddingModel model;
    std::vector<float> embedding = model.get_embeddings("Test");
    CHECK(embedding.empty());
  }

  SUBCASE("load invalid variant") {
    std::string model_dir = "embeddinggemma-300m-ONNX";
    if (!std::filesystem::exists(model_dir)) {
      return;
    }

    GemmaEmbeddingModel model;
    int result = model.load(model_dir.c_str(), "invalid_variant");
    CHECK(result != 0);
  }
}
