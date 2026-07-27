package ai.moonshine.voice;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Just enough WAV parsing to turn a reference recording into mono float PCM,
 * so {@code tts.cloneFrom("some-speech.wav")} works without the caller decoding
 * anything first.
 *
 * <p>Walks the RIFF chunks rather than assuming a 44-byte header, because real
 * recordings often carry {@code LIST} or {@code fact} chunks before the data.
 */
final class WavReader {

    static final class Audio {
        final float[] samples;
        final int sampleRate;

        Audio(float[] samples, int sampleRate) {
            this.samples = samples;
            this.sampleRate = sampleRate;
        }
    }

    private WavReader() {}

    static Audio read(File file) throws IOException {
        try (InputStream stream = new FileInputStream(file)) {
            return read(stream);
        }
    }

    static Audio read(InputStream stream) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(readAll(stream)).order(ByteOrder.LITTLE_ENDIAN);
        if (buffer.remaining() < 12 || !tagEquals(buffer, 0, "RIFF") || !tagEquals(buffer, 8, "WAVE")) {
            throw new IOException("Not a WAV file");
        }

        int format = -1;
        int channels = 1;
        int sampleRate = 0;
        int bitsPerSample = 0;
        int position = 12;
        while (position + 8 <= buffer.limit()) {
            String id = tagAt(buffer, position);
            int size = buffer.getInt(position + 4);
            int body = position + 8;
            if (size < 0 || body + size > buffer.limit()) {
                size = buffer.limit() - body;
            }
            if ("fmt ".equals(id)) {
                format = buffer.getShort(body) & 0xffff;
                channels = Math.max(1, buffer.getShort(body + 2) & 0xffff);
                sampleRate = buffer.getInt(body + 4);
                bitsPerSample = buffer.getShort(body + 14) & 0xffff;
            } else if ("data".equals(id)) {
                if (sampleRate <= 0) {
                    throw new IOException("WAV data chunk came before its format chunk");
                }
                return new Audio(decode(buffer, body, size, format, channels, bitsPerSample),
                        sampleRate);
            }
            // Chunks are word-aligned, so an odd size is followed by a pad byte.
            position = body + size + (size % 2);
        }
        throw new IOException("WAV file has no audio data");
    }

    private static float[] decode(ByteBuffer buffer, int offset, int size, int format,
            int channels, int bitsPerSample) throws IOException {
        final int bytesPerSample = bitsPerSample / 8;
        if (bytesPerSample <= 0) {
            throw new IOException("Unsupported WAV sample size: " + bitsPerSample + " bits");
        }
        final int frames = size / (bytesPerSample * channels);
        float[] samples = new float[frames];
        for (int frame = 0; frame < frames; frame++) {
            // Mono is what the models want, so mix the channels down.
            float sum = 0;
            for (int channel = 0; channel < channels; channel++) {
                int at = offset + (frame * channels + channel) * bytesPerSample;
                sum += sampleAt(buffer, at, format, bitsPerSample);
            }
            samples[frame] = sum / channels;
        }
        return samples;
    }

    private static float sampleAt(ByteBuffer buffer, int at, int format, int bitsPerSample)
            throws IOException {
        // 1 is PCM, 3 is IEEE float; 0xFFFE is WAVE_FORMAT_EXTENSIBLE, whose
        // real format lives in a sub-chunk we infer from the sample size.
        if (format == 3 && bitsPerSample == 32) {
            return buffer.getFloat(at);
        }
        switch (bitsPerSample) {
            case 8:
                return ((buffer.get(at) & 0xff) - 128) / 128f;
            case 16:
                return buffer.getShort(at) / 32768f;
            case 24: {
                int value = (buffer.get(at) & 0xff)
                        | ((buffer.get(at + 1) & 0xff) << 8)
                        | (buffer.get(at + 2) << 16);
                return value / 8388608f;
            }
            case 32:
                return format == 1 ? buffer.getInt(at) / 2147483648f : buffer.getFloat(at);
            default:
                throw new IOException("Unsupported WAV sample size: " + bitsPerSample + " bits");
        }
    }

    private static boolean tagEquals(ByteBuffer buffer, int offset, String tag) {
        return tag.equals(tagAt(buffer, offset));
    }

    private static String tagAt(ByteBuffer buffer, int offset) {
        return "" + (char) (buffer.get(offset) & 0xff) + (char) (buffer.get(offset + 1) & 0xff)
                + (char) (buffer.get(offset + 2) & 0xff) + (char) (buffer.get(offset + 3) & 0xff);
    }

    private static byte[] readAll(InputStream stream) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] chunk = new byte[64 * 1024];
        int read;
        while ((read = stream.read(chunk)) >= 0) {
            out.write(chunk, 0, read);
        }
        return out.toByteArray();
    }
}
