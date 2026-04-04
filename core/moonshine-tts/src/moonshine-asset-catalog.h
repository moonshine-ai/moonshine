#ifndef MOONSHINE_TTS_ASSET_CATALOG_H
#define MOONSHINE_TTS_ASSET_CATALOG_H

#include "file-information.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

/// Fills ``files`` with default canonical G2P paths (union of all per-language G2P dependencies).
void moonshine_asset_catalog_populate_default_g2p_files(FileInformationMap& files);

/// G2P-only asset keys for *lang_cli*, or ``std::nullopt`` if the tag is not in the catalog.
std::optional<std::vector<std::string>> moonshine_asset_catalog_g2p_dependency_keys(
    std::string_view lang_cli);

/// Sorted union of all G2P dependency keys (for ``languages`` empty / all).
std::vector<std::string> moonshine_asset_catalog_all_g2p_dependency_keys_union();

/// Tags that appear in the G2P dependency map (for TTS vocoder union and tooling).
std::vector<std::string> moonshine_asset_catalog_all_registered_language_tags();

}  // namespace moonshine_tts

#endif
