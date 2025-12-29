package ai.moonshine.voice;

import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.InstrumentationRegistry;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.logging.Logger;

import ai.moonshine.voice.JNI;
import ai.moonshine.voice.Transcript;
import ai.moonshine.voice.TranscriptLine;

public class JNITest {
    public Path tempDir;
    Logger logger;

    @Before
    public void setUp() {
        logger = Logger.getLogger(JNITest.class.getName());
        JNI.ensureLibraryLoaded();
        try {
            tempDir = Files.createTempDirectory("moonshine-jni-test");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testMoonshineGetVersion() {
        assertTrue(JNI.moonshineGetVersion() > 0);
    }

    @Test
    public void testMoonshineErrorToString() {
        assertTrue(JNI.moonshineErrorToString(JNI.MOONSHINE_ERROR_NONE) != null);
    }

    @Test
    public void testMoonshineTranscriptToString() {
        assertTrue(JNI.moonshineTranscriptToString(new Transcript()) != null);

        Transcript transcript = new Transcript();
        transcript.lines = new ArrayList<>();
        TranscriptLine line = new TranscriptLine();
        line.text = "Hello, world!";
        line.audioData = new float[100];
        line.startTime = 0.0f;
        line.duration = 1.0f;
        line.id = 0;
        line.isComplete = true;
        line.isUpdated = true;
        transcript.lines.add(line);
        String transcriptString = JNI.moonshineTranscriptToString(transcript);
        assertTrue(transcriptString != null);
        assertTrue(transcriptString.contains("Hello, world!"));
        assertTrue(transcriptString.contains("0.0"));
    }

    @Test
    public void testMoonshineLoadTranscriber() {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();

        assertTrue(
                JNI.moonshineLoadTranscriberFromFiles(null, JNI.MOONSHINE_MODEL_ARCH_TINY, null) < 0);

        byte[] encoderModelData = Utils.loadAsset(testContext, "tiny-en/encoder_model.ort");
        byte[] decoderModelData = Utils.loadAsset(testContext, "tiny-en/decoder_model_merged.ort");
        byte[] tokenizerData = Utils.loadAsset(testContext, "tiny-en/tokenizer.bin");
        assertTrue(encoderModelData != null);
        assertTrue(decoderModelData != null);
        assertTrue(tokenizerData != null);
        final int memoryTranscriberHandle = JNI.moonshineLoadTranscriberFromMemory(encoderModelData,
                decoderModelData, tokenizerData, JNI.MOONSHINE_MODEL_ARCH_TINY, null);
        assertTrue(memoryTranscriberHandle >= 0);
        JNI.moonshineFreeTranscriber(memoryTranscriberHandle);

        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/encoder_model.ort");
        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/decoder_model_merged.ort");
        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/tokenizer.bin");

        final String modelsPath = tempDir.toAbsolutePath().toString() + "/tiny-en/";

        final int filesTranscriberHandle = JNI.moonshineLoadTranscriberFromFiles(modelsPath,
                JNI.MOONSHINE_MODEL_ARCH_TINY, null);
        assertTrue(filesTranscriberHandle >= 0);
        JNI.moonshineFreeTranscriber(filesTranscriberHandle);
    }

    @Test
    public void testMoonshineTranscribe() {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/encoder_model.ort");
        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/decoder_model_merged.ort");
        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/tokenizer.bin");
        Utils.copyAssetToTempDir(testContext, tempDir, "two_cities.wav");
        final String modelsPath = tempDir.toAbsolutePath().toString() + "/tiny-en/";
        final int transcriberHandle = JNI.moonshineLoadTranscriberFromFiles(modelsPath,
                JNI.MOONSHINE_MODEL_ARCH_TINY, null);
        assertTrue(transcriberHandle >= 0);

        Utils.WavData wavData = null;
        try {
            wavData = Utils.loadWavFromAssets(testContext, "two_cities.wav");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        assertTrue(wavData.data != null);
        assertTrue(wavData.data.length > 0);
        Transcript transcription = JNI.moonshineTranscribeWithoutStreaming(transcriberHandle,
                wavData.data,
                wavData.sampleRate, 0);
        assertTrue(transcription != null);
        assertTrue(transcription.lines.size() > 0);
        StringBuilder allTextBuilder = new StringBuilder();
        for (TranscriptLine line : transcription.lines) {
            allTextBuilder.append(line.text.toLowerCase()).append(" ");
        }
        String allText = allTextBuilder.toString();
        assertTrue(allText.contains("best of times"));
        assertTrue(allText.contains("worst of times"));

        JNI.moonshineFreeTranscriber(transcriberHandle);
    }

    @Test
    public void testMoonshineStreaming() {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        Utils.WavData wavData = null;
        try {
            wavData = Utils.loadWavFromAssets(testContext, "two_cities.wav");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        assertTrue(wavData.data != null);
        assertTrue(wavData.data.length > 0);

        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/encoder_model.ort");
        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/decoder_model_merged.ort");
        Utils.copyAssetToTempDir(testContext, tempDir, "tiny-en/tokenizer.bin");
        final String modelsPath = tempDir.toAbsolutePath().toString() + "/tiny-en/";
        final int transcriberHandle = JNI.moonshineLoadTranscriberFromFiles(modelsPath,
                JNI.MOONSHINE_MODEL_ARCH_TINY, null);
        assertTrue(transcriberHandle >= 0);
        final int streamHandle = JNI.moonshineCreateStream(transcriberHandle, 0);
        assertTrue(streamHandle >= 0);
        assertTrue(JNI.moonshineStartStream(transcriberHandle,
                streamHandle) == JNI.MOONSHINE_ERROR_NONE);

        final float chunkDurationSeconds = 0.021f;
        final int chunkSize = (int) (chunkDurationSeconds * wavData.sampleRate);
        final float timeBetweenTranscriptionsSeconds = 0.387f;
        final int samplesBetweenTranscriptions = (int) (timeBetweenTranscriptionsSeconds * wavData.sampleRate);
        int samplesSinceLastTranscription = 0;
        for (int i = 0; i < wavData.data.length; i += chunkSize) {
            float[] audioData = Arrays.copyOfRange(wavData.data, i, i + chunkSize);
            assertTrue(JNI.moonshineAddAudioToStream(transcriberHandle, streamHandle, audioData,
                    wavData.sampleRate) == JNI.MOONSHINE_ERROR_NONE);

            samplesSinceLastTranscription += chunkSize;
            if (samplesSinceLastTranscription < samplesBetweenTranscriptions) {
                continue;
            }
            samplesSinceLastTranscription = 0;
            Transcript transcription = JNI.moonshineTranscribeStream(transcriberHandle, streamHandle,
                    0);
            assertTrue(transcription != null);
        }
        assertTrue(
                JNI.moonshineStopStream(transcriberHandle, streamHandle) == JNI.MOONSHINE_ERROR_NONE);

        Transcript finalTranscription = JNI.moonshineTranscribeStream(transcriberHandle, streamHandle,
                0);
        assertTrue(finalTranscription != null);
        assertTrue(finalTranscription.lines.size() > 0);
        StringBuilder allTextBuilder = new StringBuilder();
        for (TranscriptLine line : finalTranscription.lines) {
            allTextBuilder.append(line.text.toLowerCase()).append(" ");
        }
        String allText = allTextBuilder.toString();
        assertTrue(allText.contains("best of times"));
        assertTrue(allText.contains("worst of times"));

        JNI.moonshineFreeStream(transcriberHandle, streamHandle);
        JNI.moonshineFreeTranscriber(transcriberHandle);
    }
}
