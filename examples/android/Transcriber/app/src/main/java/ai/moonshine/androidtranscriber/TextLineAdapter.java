package ai.moonshine.androidtranscriber;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;
import java.util.List;

public class TextLineAdapter extends RecyclerView.Adapter<TextLineAdapter.TextLineViewHolder> {
  
  private List<String> lines;
  private OnLineClickListener onLineClickListener;
  
  public interface OnLineClickListener {
    void onLineClick(int position, String text);
  }
  
  public TextLineAdapter() {
    this.lines = new ArrayList<>();
  }
  
  public void setOnLineClickListener(OnLineClickListener listener) {
    this.onLineClickListener = listener;
  }
  
  @NonNull
  @Override
  public TextLineViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
    View view = LayoutInflater.from(parent.getContext())
        .inflate(R.layout.item_text_line, parent, false);
    return new TextLineViewHolder(view);
  }
  
  @Override
  public void onBindViewHolder(@NonNull TextLineViewHolder holder, int position) {
    String line = lines.get(position);
    holder.textView.setText(line);
    holder.itemView.setOnClickListener(v -> {
      if (onLineClickListener != null) {
        onLineClickListener.onLineClick(position, line);
      }
    });
  }
  
  @Override
  public int getItemCount() {
    return lines.size();
  }
  
  public void addLine(String text) {
    lines.add(text);
    notifyItemInserted(lines.size() - 1);
  }
  
  public void updateLastLine(String text) {
    if (!lines.isEmpty()) {
      lines.set(lines.size() - 1, text);
      notifyItemChanged(lines.size() - 1);
    }
  }
  
  public void setLines(List<String> newLines) {
    this.lines = new ArrayList<>(newLines);
    notifyDataSetChanged();
  }
  
  public List<String> getLines() {
    return new ArrayList<>(lines);
  }
  
  static class TextLineViewHolder extends RecyclerView.ViewHolder {
    TextView textView;
    
    TextLineViewHolder(@NonNull View itemView) {
      super(itemView);
      textView = itemView.findViewById(R.id.textLine);
    }
  }
}

