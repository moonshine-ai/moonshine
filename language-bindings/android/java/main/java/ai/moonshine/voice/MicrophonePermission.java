package ai.moonshine.voice;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;

import androidx.core.content.ContextCompat;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Gets {@code RECORD_AUDIO} granted before anything opens the microphone,
 * prompting the user if the system has not asked yet.
 *
 * <p>Apps used to have to declare the permission, request it themselves, and
 * then call {@code onMicPermissionGranted()} at the right point in the
 * lifecycle — three things to get right before a single word is transcribed.
 * The library now declares the permission in its own manifest and drives the
 * request through {@link MicrophonePermissionActivity}, so
 * {@link MicTranscriber#start()} just works.
 *
 * <p>{@link #ensureGranted} blocks, which is fine because the calls that need
 * it are already documented as background-thread work.
 */
final class MicrophonePermission {

    /** Long enough for a user to notice the dialog and answer it. */
    private static final long TIMEOUT_SECONDS = 120;

    private MicrophonePermission() {}

    static boolean isGranted(Context context) {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    /**
     * Returns once the microphone permission is granted, prompting if needed.
     *
     * @throws IllegalStateException if the user denies it, or does not answer.
     */
    static void ensureGranted(Context context) {
        Context appContext = context.getApplicationContext();
        if (isGranted(appContext)) {
            return;
        }

        CountDownLatch answered = new CountDownLatch(1);
        MicrophonePermissionActivity.setPendingRequest(granted -> answered.countDown());

        Intent intent = new Intent(appContext, MicrophonePermissionActivity.class);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        appContext.startActivity(intent);

        try {
            if (!answered.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                throw new IllegalStateException(
                        "Timed out waiting for the microphone permission dialog.");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted waiting for microphone permission", e);
        }

        if (!isGranted(appContext)) {
            throw new IllegalStateException(
                    "Microphone permission denied. Ask the user again from your own UI, or"
                            + " feed audio in with addAudio() instead.");
        }
    }
}
