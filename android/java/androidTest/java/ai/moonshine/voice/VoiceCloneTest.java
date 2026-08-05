package ai.moonshine.voice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Covers the clip-finding half of voice cloning. Extract now requires a loaded
 * TTS synthesizer handle; these tests load one when TTS assets are available.
 */
public class VoiceCloneTest {

    private Context context;

    @Before
    public void setUp() {
        JNI.ensureLibraryLoaded();
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
    }

    private static Utils.WavData speech() throws IOException {
        return Utils.loadWavFromAssets(
                InstrumentationRegistry.getInstrumentation().getContext(), "beckett.wav");
    }

    private int ttsHandleOrSkip() {
        TextToSpeech tts = new TextToSpeech(context).language("en");
        try {
            tts.load();
            return tts.getHandleForTests();
        } catch (RuntimeException e) {
            org.junit.Assume.assumeNoException("TTS assets unavailable", e);
            return -1;
        }
    }

    @Test
    public void testExtractsAClipFromSpeech() throws IOException {
        int handle = ttsHandleOrSkip();
        Utils.WavData wav = speech();
        SpeechClip clip = JNI.moonshineExtractSpeechClip(wav.data, wav.sampleRate, handle, 4f, 2f);
        assertNotNull(clip);
        assertTrue("expected a complete clip from real speech", clip.isComplete);
        assertNotNull(clip.audio);
        assertEquals(4 * SpeechClip.SAMPLE_RATE, clip.audio.length);
        assertTrue(clip.speechDuration >= 2f);
    }

    @Test
    public void testSilenceNeverBecomesReady() {
        int handle = ttsHandleOrSkip();
        VoiceClone clone = new VoiceClone(context, handle);
        clone.addAudio(new float[10 * VoiceClone.CLIP_SAMPLE_RATE], VoiceClone.CLIP_SAMPLE_RATE);
        assertFalse(clone.isReady());
        assertNull(clone.getAudio());
    }

    @Test
    public void testBecomesReadyAndReportsProgress() throws IOException {
        int handle = ttsHandleOrSkip();
        Utils.WavData wav = speech();
        VoiceClone clone = new VoiceClone(context, handle);
        AtomicBoolean ready = new AtomicBoolean(false);
        AtomicInteger progressCalls = new AtomicInteger(0);
        clone.onReady(() -> ready.set(true));
        clone.onProgress((recorded, speechSeconds) -> progressCalls.incrementAndGet());

        // Feed it in chunks, the way a live capture would.
        final int chunk = wav.sampleRate / 4;
        for (int offset = 0; offset < wav.data.length && !clone.isReady(); offset += chunk) {
            int length = Math.min(chunk, wav.data.length - offset);
            float[] slice = new float[length];
            System.arraycopy(wav.data, offset, slice, 0, length);
            clone.addAudio(slice, wav.sampleRate);
        }

        assertTrue("expected enough speech to be found", clone.isReady());
        assertTrue(ready.get());
        assertTrue(progressCalls.get() > 0);
        assertNotNull(clone.getAudio());
        assertEquals(VoiceClone.CLIP_SAMPLE_RATE, clone.getSampleRate());
        assertEquals(4 * VoiceClone.CLIP_SAMPLE_RATE, clone.getAudio().length);
    }

    @Test
    public void testResetClearsTheClip() throws IOException {
        int handle = ttsHandleOrSkip();
        Utils.WavData wav = speech();
        VoiceClone clone = new VoiceClone(context, handle);
        clone.addAudio(wav.data, wav.sampleRate);
        assertTrue(clone.isReady());
        clone.reset();
        assertFalse(clone.isReady());
        assertEquals(0f, clone.getRecordedSeconds(), 0.001f);
    }
}
