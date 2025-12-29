package ai.moonshine.voice;

import android.Manifest;
import android.content.pm.PackageManager;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.lang.ref.WeakReference;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * MicTranscriber handles microphone transcription with automatic permission
 * management.
 * 
 * To use without modifying your Activity's onRequestPermissionsResult, you can:
 * 1. Use the static handlePermissionResult() method in your Activity's
 * onRequestPermissionsResult
 * 2. Or poll checkPermissionStatus() periodically if you prefer not to modify
 * the Activity
 */
public class MicTranscriber {
    private static final String TAG = "MicTranscriber";
    private static final int PERMISSION_REQUEST_CODE = 1001; // Unique request code for this class

    // Static registry to map request codes to MicTranscriber instances
    private static final Map<Integer, WeakReference<MicTranscriber>> permissionRequestRegistry = new ConcurrentHashMap<>();

    private Transcriber transcriber;
    private boolean isRunning = false;
    private AppCompatActivity parentContext;
    private boolean permissionRequestPending = false;
    private boolean isMicCaptureLoopStarted = false;
    private MicCaptureProcessor micCaptureProcessor;

    public MicTranscriber(AppCompatActivity parentContext) {
        this(parentContext, "", ai.moonshine.voice.JNI.MOONSHINE_MODEL_ARCH_BASE);
    }

    public MicTranscriber(AppCompatActivity parentContext, String modelRootDir, int modelArch) {
        this.parentContext = parentContext;
        this.transcriber = new Transcriber();
        this.transcriber.loadFromAssets(parentContext, modelRootDir, modelArch);
        this.initAudioProcessing();
    }

    /**
     * Static method to handle permission results. Call this from your Activity's
     * onRequestPermissionsResult to forward permission results to MicTranscriber
     * instances.
     * 
     * This allows MicTranscriber to receive callbacks without requiring you to
     * modify
     * your Activity's onRequestPermissionsResult implementation for each instance.
     * 
     * Example usage in Activity:
     * 
     * @Override
     *           public void onRequestPermissionsResult(int requestCode, @NonNull
     *           String[] permissions,
     * @NonNull int[] grantResults) {
     *          MicTranscriber.handlePermissionResult(requestCode, permissions,
     *          grantResults);
     *          super.onRequestPermissionsResult(requestCode, permissions,
     *          grantResults);
     *          }
     */
    public static void handlePermissionResult(int requestCode, String[] permissions, int[] grantResults) {
        WeakReference<MicTranscriber> ref = permissionRequestRegistry.get(requestCode);
        if (ref != null) {
            MicTranscriber instance = ref.get();
            if (instance != null) {
                instance.onPermissionResult(grantResults);
                // Clean up after handling
                permissionRequestRegistry.remove(requestCode);
            } else {
                // Instance was garbage collected, clean up
                permissionRequestRegistry.remove(requestCode);
            }
        }
    }

    /**
     * Internal method to handle permission results for this instance.
     */
    private void onPermissionResult(int[] grantResults) {
        permissionRequestPending = false;
        if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Microphone permission granted");
            onMicPermissionGranted();
        } else {
            Log.w(TAG, "Microphone permission denied");
        }
    }

    /**
     * Request microphone permission. This will automatically handle the permission
     * request
     * and call onMicPermissionGranted() when permission is granted.
     * 
     * Note: For this to work without Activity modifications, you need to call
     * MicTranscriber.handlePermissionResult() from your Activity's
     * onRequestPermissionsResult.
     */
    public void requestMicrophonePermission() {
        if (parentContext == null) {
            Log.e(TAG, "Parent context is null, cannot request permission");
            return;
        }

        if (ContextCompat.checkSelfPermission(parentContext,
                Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Microphone permission already granted");
            onMicPermissionGranted();
            return;
        }

        permissionRequestPending = true;

        // Register this instance to receive the permission result
        permissionRequestRegistry.put(PERMISSION_REQUEST_CODE, new WeakReference<>(this));

        // Request the permission
        ActivityCompat.requestPermissions(
                parentContext,
                new String[] { Manifest.permission.RECORD_AUDIO },
                PERMISSION_REQUEST_CODE);
    }

    /**
     * Check if microphone permission has been granted. Useful for polling if you
     * prefer not to modify the Activity's onRequestPermissionsResult.
     * 
     * @return true if permission is granted, false otherwise
     */
    public boolean checkPermissionStatus() {
        if (parentContext == null) {
            return false;
        }

        boolean granted = ContextCompat.checkSelfPermission(parentContext,
                Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;

        // If permission was just granted and we were waiting for it, trigger the
        // callback
        if (granted && permissionRequestPending) {
            permissionRequestPending = false;
            onMicPermissionGranted();
        }

        return granted;
    }

    /**
     * Initialize audio processing. This will check for permission and request it if
     * needed.
     */
    public void initAudioProcessing() {
        requestMicrophonePermission();
    }

    /**
     * Called when microphone permission is granted. Override this or call it to
     * continue
     * with audio processing setup.
     */
    protected void onMicPermissionGranted() {
        CompletableFuture.runAsync(this::startMicCaptureLoop).thenRun(this::startAudioProcessingLoop);
    }

    private void startMicCaptureLoop() {
        if (isMicCaptureLoopStarted) {
            return;
        }
        isMicCaptureLoopStarted = true;
        micCaptureProcessor = new MicCaptureProcessor();
        Thread micThread = new Thread(micCaptureProcessor);
        micThread.start();
    }

    private void startAudioProcessingLoop() {
        Thread audioProcessingThread = new Thread(new Runnable() {
            @Override
            public void run() {
                audioProcessingLoop();
            }
        });
        audioProcessingThread.start();
    }

    public void stop() {
        this.isRunning = false;
    }

    public void start() {
        this.isRunning = true;
    }

    /**
     * Get the permission request code used by this class.
     * Useful for Activities that need to check if a permission request belongs to
     * this class.
     */
    public static int getPermissionRequestCode() {
        return PERMISSION_REQUEST_CODE;
    }

    private void audioProcessingLoop() {
        int streamHandle = this.transcriber.createStream();
        this.transcriber.startStream(streamHandle);
        this.isRunning = true;
        boolean wasRunning = this.isRunning;
        while (!Thread.currentThread().isInterrupted()) {
            float[] audioData = micCaptureProcessor.consumeAudio();
            if (!this.isRunning && !wasRunning) {
                continue;
            }
            if (this.isRunning && !wasRunning) {
                this.transcriber.startStream(streamHandle);
            }
            if (this.isRunning || wasRunning) {
                this.transcriber.addAudioToStream(streamHandle, audioData, 16000);
            }
            if (!this.isRunning && wasRunning) {
                this.transcriber.stopStream(streamHandle);
            }
            wasRunning = this.isRunning;
        }
        this.transcriber.stopStream(streamHandle);
        this.transcriber.freeStream(streamHandle);
    }
}
