package ai.moonshine.voice;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Utils {

    /**
     * Log median, mean, min, and max stats for a list of values.
     */
    public static void logStats(String tag, String label, List<Long> values) {
        if (values.isEmpty()) {
            Log.i(tag, label + ": [no values]");
            return;
        }

        // Sort for median
        List<Long> sorted = new ArrayList<>(values);
        Collections.sort(sorted);

        long min = sorted.get(0);
        long max = sorted.get(sorted.size() - 1);

        // Mean
        long sum = 0;
        for (long v : sorted) sum += v;
        double mean = sum / (double) sorted.size();

        // Median
        double median;
        int n = sorted.size();
        if (n % 2 == 1) {
            median = sorted.get(n / 2);
        } else {
            median = (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
        }

        Log.i(tag, String.format(
                "%s → min=%d ms, max=%d ms, median=%.2f ms, mean=%.2f ms (n=%d)",
                label, min, max, median, mean, n
        ));
    }

    /**
     * Remove all Unicode punctuation from a string.
     */
    public static String removePunctuation(String input) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < input.length(); i++) {
            int type = Character.getType(input.charAt(i));
            if (type != Character.CONNECTOR_PUNCTUATION &&
                    type != Character.DASH_PUNCTUATION &&
                    type != Character.START_PUNCTUATION &&
                    type != Character.END_PUNCTUATION &&
                    type != Character.INITIAL_QUOTE_PUNCTUATION &&
                    type != Character.FINAL_QUOTE_PUNCTUATION &&
                    type != Character.OTHER_PUNCTUATION) {
                sb.append(input.charAt(i));
            }
        }
        return sb.toString();
    }

    /**
     * Compute normalized Levenshtein edit distance between two strings.
     * Strings are lowercased and punctuation is removed.
     * Result is between 0.0 (identical) and 1.0 (completely different).
     */
    public static double normalizedEditDistance(String s1, String s2) {
        String a = removePunctuation(s1.toLowerCase());
        String b = removePunctuation(s2.toLowerCase());

        int n = a.length();
        int m = b.length();

        if (n == 0 && m == 0) return 0.0;
        if (n == 0) return (double) m / Math.max(n, m);
        if (m == 0) return (double) n / Math.max(n, m);

        // DP array (space optimized)
        int[] prev = new int[m + 1];
        for (int j = 0; j <= m; j++) {
            prev[j] = j;
        }

        for (int i = 1; i <= n; i++) {
            int[] cur = new int[m + 1];
            cur[0] = i;
            for (int j = 1; j <= m; j++) {
                int cost = (a.charAt(i - 1) == b.charAt(j - 1)) ? 0 : 1;
                cur[j] = Math.min(
                        Math.min(prev[j] + 1,       // deletion
                                cur[j - 1] + 1),   // insertion
                        prev[j - 1] + cost           // substitution
                );
            }
            prev = cur;
        }

        int distance = prev[m];
        return (double) distance / Math.max(n, m);
    }

    public static class WavData {
        public final float[] data;
        public final int sampleRate;

        public WavData(float[] data, int sampleRate) {
            this.data = data;
            this.sampleRate = sampleRate;
        }
    }

    /**
     * Load 16-bit PCM WAV from assets into float array [-1,1]
     */
    public static WavData loadWavFromAssets(Context context, String assetPath) throws IOException {
        AssetManager am = context.getAssets();
        InputStream is = am.open(assetPath);

        byte[] header = new byte[44];
        if (is.read(header) != 44) {
            is.close();
            throw new IOException("Invalid WAV header");
        }

        // Check "RIFF" and "WAVE"
        if (header[0] != 'R' || header[1] != 'I' || header[2] != 'F' || header[3] != 'F' ||
                header[8] != 'W' || header[9] != 'A' || header[10] != 'V' || header[11] != 'E') {
            is.close();
            throw new IOException("Not a valid WAV file");
        }

        int bitsPerSample = ((header[34] & 0xff) | ((header[35] & 0xff) << 8));
        int sampleRate = ((header[24] & 0xff) | ((header[25] & 0xff) << 8) |
                ((header[26] & 0xff) << 16) | ((header[27] & 0xff) << 24));
        int dataSize = ((header[40] & 0xff) | ((header[41] & 0xff) << 8) |
                ((header[42] & 0xff) << 16) | ((header[43] & 0xff) << 24));

        if (bitsPerSample != 16) {
            is.close();
            throw new IOException("Only 16-bit PCM WAV supported");
        }

        byte[] dataBytes = new byte[dataSize];
        if (is.read(dataBytes) != dataSize) {
            is.close();
            throw new IOException("Could not read WAV data");
        }
        is.close();

        int numSamples = dataSize / 2;
        float[] samples = new float[numSamples];

        ByteBuffer bb = ByteBuffer.wrap(dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < numSamples; i++) {
            short val = bb.getShort();
            samples[i] = val / 32768.0f;
        }

        return new WavData(samples, sampleRate);
    }
    public static byte[] loadAsset(Context context, String path) {
        try {
            InputStream is = context.getAssets().open(path);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            return buffer;
        } catch (IOException e) {
            return null;
        }
    }

    public static void copyAssetToTempDir(Context context, Path tempDir, String assetPath) {
        try {
            InputStream is = context.getAssets().open(assetPath);
            Path tempDirPath = tempDir.toAbsolutePath();
            Path targetPath = tempDirPath.resolve(assetPath);
            Files.createDirectories(targetPath.getParent());
            FileOutputStream fos = new FileOutputStream(targetPath.toFile());
            byte[] buffer = new byte[4096];
            int i;
            while ((i = is.read(buffer)) != -1) {
                fos.write(buffer, 0, i);
            }
            fos.close();
            is.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
