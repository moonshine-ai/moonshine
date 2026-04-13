package ai.moonshine.examples.intentrecognizer

import android.content.Context
import java.io.File
import java.io.FileOutputStream

/** Recursively copies one directory tree from assets to a filesystem directory. */
internal object AssetDirectoryCopy {
    /** Copies [assetDir] into [destRoot] when [readyCheck] is missing under [destRoot]. */
    fun copyDirIfNeeded(
        context: Context,
        assetDir: String,
        destRoot: File,
        readyCheckRelative: String,
    ) {
        val ready = File(destRoot, readyCheckRelative)
        if (ready.isFile) {
            return
        }
        destRoot.deleteRecursively()
        copyRecursive(context, assetDir, destRoot)
    }

    private fun copyRecursive(context: Context, assetPath: String, destDir: File) {
        val names = context.assets.list(assetPath) ?: return
        destDir.mkdirs()
        for (name in names) {
            val subAsset = "$assetPath/$name"
            val children = context.assets.list(subAsset)
            val target = File(destDir, name)
            if (children.isNullOrEmpty()) {
                context.assets.open(subAsset).use { input ->
                    FileOutputStream(target).use { output -> input.copyTo(output) }
                }
            } else {
                copyRecursive(context, subAsset, target)
            }
        }
    }
}
