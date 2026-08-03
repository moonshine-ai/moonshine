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

    /**
     * Live audio arrives in capture buffers of a few tens of milliseconds, but a
     * transcription pass is expensive and grows more so as a session runs, so
     * addAudio must coalesce them rather than transcribe once per buffer.
     */
    @Test
    public void testAddAudioTranscribesPerIntervalNotPerChunk() {
        Transcriber transcriber = new Transcriber();
        final int sampleRate = 16000;
        final int streamHandle = 0;

        // A capture buffer's worth at a time, the way MicCaptureProcessor reads.
        final int chunkSamples = 256;
        final int chunks = 1000;
        int passes = 0;
        for (int i = 0; i < chunks; i++) {
            if (transcriber.isUpdateDue(streamHandle, chunkSamples, sampleRate)) {
                passes += 1;
            }
        }

        double seconds = (double) (chunks * chunkSamples) / sampleRate;
        int expected = (int) (seconds / Transcriber.DEFAULT_UPDATE_INTERVAL);
        assertTrue("expected about " + expected + " passes over " + seconds
                        + "s of audio, got " + passes,
                Math.abs(passes - expected) <= 1);
        assertTrue("gate should be well below one pass per chunk", passes < chunks / 10);
    }

    /**
     * The interval is a floor rather than a cadence: a pass has to cover at
     * least as much audio as the last one took to make. Most of what a pass
     * costs is not the audio in it -- measured on the tiny model with speakers,
     * 102ms of a pass goes on getting started and 269ms on each second of audio
     * it looks at -- so asking twice a second pays that overhead twice a second,
     * and a machine that cannot quite afford it does not fall behind by a fixed
     * amount, it falls behind further every pass. Making a pass earn its keep
     * turns that into batch behaviour instead.
     *
     * <p>What a pass cost is said here rather than taken, so that none of this
     * depends on how long anything really takes.
     */
    @Test
    public void testAPassMustCoverAsMuchAudioAsTheLastOneCost() {
        Transcriber transcriber = new Transcriber();
        final int sampleRate = 16000;
        final int streamHandle = 3;
        // An eighth of a second, which is exact in binary and so cannot leave a
        // sum of chunks a hair under the amount being waited for.
        final int eighthOfASecond = sampleRate / 8;

        // Two seconds a pass, which is four intervals' worth of audio.
        transcriber.lastPassSeconds.put(streamHandle, 2.0);

        int passes = 0;
        for (int i = 0; i < 32; i++) {
            if (transcriber.isUpdateDue(streamHandle, eighthOfASecond, sampleRate)) {
                passes += 1;
            }
        }
        // Four seconds of audio is two passes at two seconds each, where the
        // floor alone would have asked for eight.
        assertTrue("four seconds at two seconds a pass should be two passes, got "
                        + passes,
                passes == 2);

        // With time to spare, nothing changes.
        transcriber.lastPassSeconds.put(streamHandle, 0.05);
        passes = 0;
        for (int i = 0; i < 32; i++) {
            if (transcriber.isUpdateDue(streamHandle, eighthOfASecond, sampleRate)) {
                passes += 1;
            }
        }
        assertTrue("four seconds should be about eight passes at the floor, got "
                        + passes,
                Math.abs(passes - 8) <= 1);
    }

    /**
     * One freak pass -- a collection, a phone that went to sleep mid-call --
     * must not leave a live transcript silent for a minute afterwards.
     */
    @Test
    public void testTheWaitAfterASlowPassIsCapped() {
        Transcriber transcriber = new Transcriber();
        final int sampleRate = 16000;
        final int streamHandle = 4;
        final int second = sampleRate;

        transcriber.lastPassSeconds.put(streamHandle, 60.0);

        int passes = 0;
        for (int i = 0; i < 6; i++) {
            if (transcriber.isUpdateDue(streamHandle, second, sampleRate)) {
                passes += 1;
            }
        }
        // Ten intervals is five seconds, so six seconds of audio is one pass,
        // not the none a minute-long wait would have allowed.
        assertTrue("a capped wait should still let a pass through, got " + passes,
                passes == 1);
    }

    /** The gate counts audio, so chunk size cannot change how often it fires. */
    @Test
    public void testUpdateIntervalIsMeasuredInAudioNotCalls() {
        Transcriber transcriber = new Transcriber();
        final int sampleRate = 16000;
        final int seconds = 10;

        int smallChunkPasses = 0;
        for (int i = 0; i < seconds * sampleRate / 128; i++) {
            if (transcriber.isUpdateDue(0, 128, sampleRate)) {
                smallChunkPasses += 1;
            }
        }

        int largeChunkPasses = 0;
        for (int i = 0; i < seconds * sampleRate / 2048; i++) {
            if (transcriber.isUpdateDue(1, 2048, sampleRate)) {
                largeChunkPasses += 1;
            }
        }

        // 1250 calls and 78 calls carry the same ten seconds, so they owe the
        // same handful of passes, give or take where the last remainder lands.
        assertTrue("chunk size changed the pass count: " + smallChunkPasses
                        + " vs " + largeChunkPasses,
                Math.abs(smallChunkPasses - largeChunkPasses) <= 1);
    }

    /** Two streams running at once must not spend each other's audio. */
    @Test
    public void testUpdateIntervalIsTrackedPerStream() {
        Transcriber transcriber = new Transcriber();
        final int sampleRate = 16000;
        final int quarterSecond = sampleRate / 4;

        assertTrue(!transcriber.isUpdateDue(7, quarterSecond, sampleRate));
        assertTrue(!transcriber.isUpdateDue(8, quarterSecond, sampleRate));
        // Only stream 7 has half a second in it now, so only it is due.
        assertTrue(transcriber.isUpdateDue(7, quarterSecond, sampleRate));
        assertTrue(!transcriber.isUpdateDue(8, 1, sampleRate));
    }

    /** Setting the interval to zero restores a pass for every call. */
    @Test
    public void testZeroUpdateIntervalTranscribesEveryCall() {
        Transcriber transcriber = new Transcriber();
        transcriber.setUpdateInterval(0.0);
        for (int i = 0; i < 10; i++) {
            assertTrue(transcriber.isUpdateDue(0, 128, 16000));
        }
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
