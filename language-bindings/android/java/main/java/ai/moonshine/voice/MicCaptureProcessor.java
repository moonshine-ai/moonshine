package ai.moonshine.voice;

import android.media.AudioFormat;
import android.media.MediaRecorder;
import android.media.AudioRecord;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Reads a stream of audio data from the microphone on a separate thread, processing it into
 * a list of MicCaptureChunks that can be accessed via consumeChunks()
 */
public class MicCaptureProcessor implements Runnable {
    protected ReentrantLock mutex = new ReentrantLock();
    protected final List<float[]> audioChunks = new CopyOnWriteArrayList<>();
    private final int bufferSize = 512;

    public float[] consumeAudio() {
        mutex.lock();
        List<float[]> chunks = new ArrayList<>(audioChunks);
        audioChunks.clear();
        mutex.unlock();
        int totalLength = 0;
        for (float[] chunk : chunks) {
            totalLength += chunk.length;
        }
        float[] audio = new float[totalLength];
        int offset = 0;
        for (float[] chunk : chunks) {
            System.arraycopy(chunk, 0, audio, offset, chunk.length);
            offset += chunk.length;
        }
        return audio;
    }

    @Override
    public void run() {
        // Microphone audio capture at 16KHz, PCM 16-bit, mono
        final int sampleRate = 16000;
        final int channelConfig = AudioFormat.CHANNEL_IN_MONO;
        final int audioFormat = AudioFormat.ENCODING_PCM_16BIT;
        final int minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);

        // AudioRecord instance
        AudioRecord audioRecord = null;
        try {
            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channelConfig,
                    audioFormat,
                    minBufferSize);
        } catch (SecurityException e) {
            Log.e("MicCaptureProcessor", "Microphone permission not granted: " + e.getMessage());
            return;
        }

        audioRecord.startRecording();
        while (!Thread.currentThread().isInterrupted()) {
            // Create a new buffer each time, rather than reusing the same buffer, so that
            // we avoid issues with shallow copies being made, and overwritten by subsequent
            // reads before they're accessed by the audio processing pipeline.
            byte[] audioBuffer = new byte[bufferSize];
            int read = audioRecord.read(audioBuffer, 0, audioBuffer.length);

            if (read > 0 && read % 2 == 0) { // Ensure we have a whole number of shorts
                mutex.lock();
                // VAD takes 16-bit raw audio in audioBuffer
                // Transcriber takes PCM audio in pcmAudio
                int numShorts = audioBuffer.length / 2;
                float[] pcmAudio = new float[numShorts];
                for (int i = 0; i < numShorts; i++) {
                    // Convert two bytes (little-endian) to a short
                    short pcmSample = (short) ((audioBuffer[2 * i + 1] << 8) | (audioBuffer[2 * i] & 0xff));
                    // Convert short PCM to float in range -1.0 to 1.0
                    pcmAudio[i] = pcmSample / 32768.0f;
                }
                audioChunks.add(pcmAudio);
                mutex.unlock();
            }
        }
        audioRecord.stop();
        audioRecord.release();
    }
}
