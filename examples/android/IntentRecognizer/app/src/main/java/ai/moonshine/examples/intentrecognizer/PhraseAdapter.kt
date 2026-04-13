package ai.moonshine.examples.intentrecognizer

import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import ai.moonshine.examples.intentrecognizer.databinding.ItemPhraseRowBinding

internal data class PhraseRow(val id: Long, var text: String)

internal class PhraseAdapter(
    private val onScheduleIntentSync: () -> Unit,
    private val onRemoveRow: (Long) -> Unit,
) : RecyclerView.Adapter<PhraseAdapter.VH>() {

    private val items = mutableListOf<PhraseRow>()
    private val highlighted = mutableSetOf<Long>()

    init {
        setHasStableIds(true)
    }

    fun resetToDefaults(phrases: List<String>) {
        items.clear()
        var nid = 1L
        for (p in phrases) {
            items.add(PhraseRow(nid++, p))
        }
        notifyDataSetChanged()
    }

    override fun getItemId(position: Int): Long = items[position].id

    override fun getItemCount(): Int = items.size

    fun addEmptyRow() {
        val nid = (items.maxOfOrNull { it.id } ?: 0L) + 1L
        items.add(PhraseRow(nid, ""))
        notifyItemInserted(items.size - 1)
    }

    fun removeRow(id: Long) {
        val idx = items.indexOfFirst { it.id == id }
        if (idx >= 0) {
            items.removeAt(idx)
            notifyItemRemoved(idx)
        }
    }

    fun flashHighlight(id: Long) {
        highlighted.add(id)
        val idx = items.indexOfFirst { it.id == id }
        if (idx >= 0) {
            notifyItemChanged(idx)
        }
        Handler(Looper.getMainLooper()).postDelayed({
            highlighted.remove(id)
            val j = items.indexOfFirst { it.id == id }
            if (j >= 0) {
                notifyItemChanged(j)
            }
        }, 1800L)
    }

    /** Returns registered text for each row in order (for intent registration). */
    fun currentPhrases(): List<String> = items.map { it.text.trim() }

    fun rowIdMatchingCanonical(canonical: String): Long? {
        val c = canonical.trim()
        return items.firstOrNull { it.text.trim() == c }?.id
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val binding = ItemPhraseRowBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return VH(binding)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        holder.bind(items[position])
    }

    inner class VH(private val binding: ItemPhraseRowBinding) : RecyclerView.ViewHolder(binding.root) {
        private var watcher: TextWatcher? = null

        fun bind(row: PhraseRow) {
            binding.phraseEdit.removeTextChangedListener(watcher)
            binding.phraseEdit.setText(row.text)
            val bg = if (highlighted.contains(row.id)) {
                ContextCompat.getColor(binding.root.context, R.color.highlight_background)
            } else {
                Color.TRANSPARENT
            }
            binding.phraseRowRoot.setBackgroundColor(bg)

            watcher = object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                override fun afterTextChanged(s: Editable?) {
                    row.text = s?.toString().orEmpty()
                    onScheduleIntentSync()
                }
            }
            binding.phraseEdit.addTextChangedListener(watcher)

            binding.removePhraseButton.setOnClickListener {
                onRemoveRow(row.id)
            }
        }
    }
}
