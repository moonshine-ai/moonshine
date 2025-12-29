package ai.moonshine.voice;

import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Files;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import ai.moonshine.voice.JNI;
import ai.moonshine.voice.TranscriptEvent;
import ai.moonshine.voice.TranscriptEventListener;
import ai.moonshine.voice.Transcriber;
import ai.moonshine.voice.TranscriptLine;
import ai.moonshine.voice.TranscriberOption;

public class NoTranscriptionTest {
    private Logger logger = Logger.getLogger(NoTranscriptionTest.class.getName());
    private Path tempDir;
    private int startedCount = 0;
    private int updatedCount = 0;
    private int completedCount = 0;
    private int textChangedCount = 0;

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
    public void testMoonshineNoTranscription() {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        Utils.WavData wavData = null;
        try {
            wavData = Utils.loadWavFromAssets(testContext, "two_cities.wav");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        assertTrue(wavData.data != null);
        assertTrue(wavData.data.length > 0);

        List<TranscriberOption> options = new ArrayList<>();
        options.add(new TranscriberOption("skip_transcription", "true"));
        Transcriber transcriber = new Transcriber(options);
        transcriber.loadFromFiles(null, JNI.MOONSHINE_MODEL_ARCH_TINY);
        transcriber.start();

        startedCount = 0;
        updatedCount = 0;
        completedCount = 0;
        textChangedCount = 0;

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
        assertTrue(textChangedCount == 0);
    }

    public void onLineStartedEvent(TranscriptLine line) {
        logger.log(Level.INFO, "Transcription started: " + line.toString());
        assertTrue(line.text == null);
        assertTrue(line.isNew);
        assertTrue(line.isUpdated);
        assertTrue(line.hasTextChanged == false);
        startedCount += 1;
    }

    public void onLineUpdatedEvent(TranscriptLine line) {
        assertTrue(line.isUpdated);
        assertTrue(!line.isNew);
        assertTrue(!line.isComplete);
        assertTrue(line.hasTextChanged == false);
        updatedCount += 1;
    }

    public void onLineTextChangedEvent(TranscriptLine line) {
        // This should never happen.
        assertTrue(false);
        textChangedCount += 1;
    }

    public void onLineCompletedEvent(TranscriptLine line) {
        logger.log(Level.INFO, "Transcription line completed: " + line.toString());
        assertTrue(line.isComplete);
        assertTrue(line.isUpdated);
        assertTrue(line.hasTextChanged == false);
        assertTrue(line.text == null);
        completedCount += 1;
    }
}
