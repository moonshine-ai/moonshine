package ai.moonshine.androidtranscriber;

import ai.moonshine.androidtranscriber.databinding.ActivityMainBinding;
import ai.moonshine.voice.JNI;
import ai.moonshine.voice.MicTranscriber;
import ai.moonshine.voice.TranscriptEvent;
import ai.moonshine.voice.TranscriptEventListener;

import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.EditText;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

  private AppBarConfiguration appBarConfiguration;
  private ActivityMainBinding binding;
  private RecyclerView messagesRecyclerView;
  private TextLineAdapter adapter;
  private FloatingActionButton fab;
  private TextView fabPressMessage;

  private MicTranscriber transcriber;
  private boolean isTranscribing = false;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    binding = ActivityMainBinding.inflate(getLayoutInflater());
    setContentView(binding.getRoot());

    messagesRecyclerView = findViewById(R.id.messagesRecyclerView);
    adapter = new TextLineAdapter();
    messagesRecyclerView.setLayoutManager(new LinearLayoutManager(this));
    messagesRecyclerView.setAdapter(adapter);

    transcriber = new MicTranscriber(this);
    transcriber.loadFromAssets(this, "base-en", JNI.MOONSHINE_MODEL_ARCH_BASE);
    transcriber.addListener(
        event -> event.accept(new TranscriptEventListener() {
          @Override
          public void onLineStarted(TranscriptEvent.LineStarted e) {
            runOnUiThread(() -> {
              adapter.addLine("...");
              messagesRecyclerView.smoothScrollToPosition(
                  adapter.getItemCount() - 1);
            });
          }
          @Override
          public void onLineTextChanged(TranscriptEvent.LineTextChanged e) {
            runOnUiThread(() -> {
              adapter.updateLastLine(e.line.text);
              messagesRecyclerView.smoothScrollToPosition(
                  adapter.getItemCount() - 1);
            });
          }
          @Override
          public void onLineCompleted(TranscriptEvent.LineCompleted e) {
            runOnUiThread(() -> {
              adapter.updateLastLine(e.line.text);
              messagesRecyclerView.smoothScrollToPosition(
                  adapter.getItemCount() - 1);
            });
          }
        }));

    // Request microphone permissions if not already granted
    if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) ==
        PackageManager.PERMISSION_GRANTED) {
      transcriber.onMicPermissionGranted();
    } else {
      requestPermissions(
          new String[] {android.Manifest.permission.RECORD_AUDIO},
          1 // Request code
      );
    }

    fab = findViewById(R.id.fab);
    fab.setBackgroundTintList(android.content.res.ColorStateList.valueOf(
        android.graphics.Color.parseColor("#ffffff")));
    fab.setOnClickListener(v -> {
      fabPressMessage.setVisibility(View.GONE);
      if (isTranscribing) {
        transcriber.stop();
        isTranscribing = false;
        fab.setBackgroundTintList(android.content.res.ColorStateList.valueOf(
            android.graphics.Color.parseColor("#ffffff")));
      } else {
        transcriber.start();
        isTranscribing = true;
        fab.setBackgroundTintList(android.content.res.ColorStateList.valueOf(
            android.graphics.Color.parseColor("#aaaaff")));
      }
    });
    fabPressMessage = findViewById(R.id.fabPressMessage);
    ObjectAnimator animator =
        ObjectAnimator.ofFloat(fabPressMessage, "translationY", -30f, -70f);
    animator.setDuration(1000);
    animator.setRepeatCount(ValueAnimator.INFINITE);
    animator.setRepeatMode(ValueAnimator.REVERSE);
    animator.start();
  }

  @Override
  public void onRequestPermissionsResult(int requestCode,
                                         @NonNull String[] permissions,
                                         @NonNull int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == 1) {
      if (grantResults.length > 0 &&
          grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        transcriber.onMicPermissionGranted();
      }
    }
  }
}