package ai.moonshine.voice;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.concurrent.atomic.AtomicReference;

/**
 * An invisible activity whose only job is to show the {@code RECORD_AUDIO}
 * permission dialog and report the answer back to {@link MicrophonePermission}.
 *
 * <p>Android only delivers a permission result to an activity, so a library
 * that wants to own the prompt has to bring its own. It is transparent, has no
 * layout, and finishes as soon as the user answers, so from the application's
 * point of view nothing happened except the system dialog.
 */
public final class MicrophonePermissionActivity extends Activity {

    /** Notified with the user's answer. */
    interface Listener {
        void onResult(boolean granted);
    }

    private static final AtomicReference<Listener> PENDING = new AtomicReference<>();
    private static final int REQUEST_CODE = 0x4d53;  // 'MS'

    static void setPendingRequest(Listener listener) {
        PENDING.set(listener);
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestPermissions(new String[] {Manifest.permission.RECORD_AUDIO}, REQUEST_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
            @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != REQUEST_CODE) {
            return;
        }
        boolean granted = grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED;
        finishWith(granted);
    }

    @Override
    protected void onDestroy() {
        // Covers the user dismissing the dialog with the back gesture, which
        // never reaches onRequestPermissionsResult and would otherwise leave
        // the waiting thread parked until it times out.
        finishWith(false);
        super.onDestroy();
    }

    private void finishWith(boolean granted) {
        Listener listener = PENDING.getAndSet(null);
        if (listener != null) {
            listener.onResult(granted);
        }
        if (!isFinishing()) {
            finish();
        }
    }
}
