package ai.moonshine.voice;

import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.util.Log;

import androidx.test.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Files;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import ai.moonshine.voice.JNI;
import ai.moonshine.voice.Transcriber;
import ai.moonshine.voice.Transcript;
import ai.moonshine.voice.TranscriptEvent;
import ai.moonshine.voice.TranscriptEventListener;
import ai.moonshine.voice.TranscriptLine;

public class TranscriberTest {
    private Logger logger = Logger.getLogger(TranscriberTest.class.getName());
    private Path tempDir;
    private int startedCount = 0;
    private int updatedCount = 0;
    private int completedCount = 0;
    private int textChangedCount = 0;
    private StringBuilder allTextBuilder;
    private Map<Long, TranscriptLine> previousTranscriptLines;

    @Before
    public void setUp() {
        JNI.ensureLibraryLoaded();
        try {
            tempDir = Files.createTempDirectory("voice-transcriber-test");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    public void testMoonshineTranscriberStreaming() {
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

        Transcriber transcriber = new Transcriber();
        transcriber.loadFromFiles(modelsPath, JNI.MOONSHINE_MODEL_ARCH_TINY);
        transcriber.start();

        startedCount = 0;
        updatedCount = 0;
        completedCount = 0;
        textChangedCount = 0;
        previousTranscriptLines = new HashMap<>();

        allTextBuilder = new StringBuilder();

        transcriber.addListener(event -> event.accept(new TranscriptEventListener() {
            @Override
            public void onLineStarted(TranscriptEvent.LineStarted e) {
                onLineStartedEvent(e.line);
            }

            @Override
            public void onLineUpdated(TranscriptEvent.LineUpdated e) {
                onLineUpdatedEvent(e.line);
            }

            @Override
            public void onLineTextChanged(TranscriptEvent.LineTextChanged e) {
                onLineTextChangedEvent(e.line);
            }

            @Override
            public void onLineCompleted(TranscriptEvent.LineCompleted e) {
                onLineCompletedEvent(e.line);
            }

            @Override
            public void onError(TranscriptEvent.Error e) {
                logger.log(Level.INFO, "Transcription error: {}", e.cause.getMessage());
                assertTrue("Transcription error: " + e.cause.getMessage(), false);
            }
        }));

        final float chunkDurationSeconds = 0.017f;
        final int chunkSize = (int) (chunkDurationSeconds * wavData.sampleRate);
        for (int i = 0; i < wavData.data.length; i += chunkSize) {
            float[] audioData = Arrays.copyOfRange(wavData.data, i, i + chunkSize);
            transcriber.addAudio(audioData, wavData.sampleRate);
        }
        transcriber.stop();
        assertTrue(startedCount > 0);
        assertTrue(updatedCount > 0);
        assertTrue(completedCount > 0);
        assertTrue(startedCount == completedCount);
        assertTrue(updatedCount >= startedCount);
        assertTrue(textChangedCount > 0);
        String allText = allTextBuilder.toString().toLowerCase();
        assertTrue(allText.contains("best of times"));
        assertTrue(allText.contains("worst of times"));
    }

    public void onLineStartedEvent(TranscriptLine line) {
        logger.log(Level.INFO, "Transcription started: " + line.toString());
        assertTrue(line.isNew);
        assertTrue(line.isUpdated);
        assertTrue(previousTranscriptLines.get(line.id) == null);
        startedCount += 1;
    }

    public void onLineUpdatedEvent(TranscriptLine line) {
        assertTrue(line.isUpdated);
        assertTrue(!line.isNew);
        assertTrue(!line.isComplete);
        updatedCount += 1;
    }

    public void onLineTextChangedEvent(TranscriptLine line) {
        assertTrue(line.hasTextChanged);
        TranscriptLine previousLine = previousTranscriptLines.get(line.id);
        if (previousLine == null) {
            previousTranscriptLines.put(line.id, line);
        } else {
            assertTrue(!previousLine.text.equals(line.text));
            previousTranscriptLines.put(line.id, line);
        }
        textChangedCount += 1;
    }

    public void onLineCompletedEvent(TranscriptLine line) {
        logger.log(Level.INFO, "Transcription line completed: " + line.toString());
        assertTrue(line.isComplete);
        assertTrue(line.isUpdated);
        assertTrue(previousTranscriptLines.get(line.id) != null);
        assertTrue(previousTranscriptLines.get(line.id).text.equals(line.text));
        completedCount += 1;
        allTextBuilder.append(line.text).append("\n");
    }

    @Test
    public void testMoonshineTranscriberWithoutStreaming() {
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

        Transcriber transcriber = new Transcriber();
        transcriber.loadFromFiles(modelsPath, JNI.MOONSHINE_MODEL_ARCH_TINY);

        final Transcript transcript = transcriber.transcribeWithoutStreaming(wavData.data, wavData.sampleRate);
        assertTrue(transcript != null);
        assertTrue(transcript.lines.size() > 0);
        StringBuilder allTextBuilder = new StringBuilder();
        for (TranscriptLine line : transcript.lines) {
            assertTrue(line.isNew);
            assertTrue(line.isUpdated);
            assertTrue(line.hasTextChanged);
            assertTrue(line.isComplete);
            allTextBuilder.append(line.text.toLowerCase()).append(" ");
        }
        String allText = allTextBuilder.toString();
        assertTrue(allText.contains("best of times"));
        assertTrue(allText.contains("worst of times"));
    }
}
