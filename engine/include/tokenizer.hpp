#pragma once

#include <cstdint>
#include <array>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace curriculum {

/// Lossless byte-level BPE tokenizer for curriculum-trained vocabularies.
///
/// The Rust trainer and this C++ inference implementation share:
/// - the same deterministic pre-tokenization rules,
/// - numeric merge rules (left ID, right ID, result ID),
/// - byte-exact decoding.
class Tokenizer {
public:
    Tokenizer() = default;

    bool load(const std::string& vocab_path, const std::string& merges_path);
    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    std::vector<std::vector<int32_t>> encode_batch(
        const std::vector<std::string>& texts, size_t num_threads = 0) const;

    size_t vocab_size() const;
    void set_cache_capacity(size_t capacity);
    void clear_cache();
    std::unordered_map<std::string, size_t> cache_info() const;

    /// Safe, printable token representation used by vocab.json.
    std::string id_to_token(int32_t id) const;

    /// Find an ID from literal token text/bytes. Returns -1 when absent.
    int32_t token_to_id(const std::string& token) const;

    void set_special_tokens(const std::vector<std::string>& tokens);
    void configure_special_tokens(
        const std::unordered_map<std::string, int32_t>& token_ids,
        const std::unordered_map<int32_t, std::string>& aliases,
        size_t expected_vocab_size);
    // Optional decode-only aliases explicitly declared by legacy artifacts.
    void set_special_token_aliases(const std::unordered_map<int32_t, std::string>& aliases);

private:
    struct EncodingScratch;
    struct CachedPiece {
        std::string bytes;
        std::vector<int32_t> ids;
    };
    struct CacheShard {
        mutable std::mutex mutex;
        std::list<CachedPiece> lru;
        // Views refer to owned, stable strings in the LRU list.
        std::unordered_map<std::string_view, std::list<CachedPiece>::iterator> index;
        size_t payload_bytes = 0;
        size_t hits = 0;
        size_t misses = 0;
    };
    static constexpr size_t kCacheShards = 16;
    static constexpr size_t kCachePayloadBytes = 4 * 1024 * 1024;
    static constexpr size_t kMaxCachedPieceBytes = 256;
    mutable std::shared_mutex model_mutex_;
    mutable std::array<CacheShard, kCacheShards> piece_cache_;
    size_t cache_capacity_ = 16384;

    struct MergeInfo {
        size_t rank;
        int32_t result;
    };

    std::vector<std::string> id_to_token_; // raw token bytes
    std::unordered_map<std::string, int32_t> token_to_id_; // raw bytes -> ID

    // Ordered merge rules: left, right, result.
    std::vector<std::tuple<int32_t, int32_t, int32_t>> merges_;
    // Earliest rule for each pair. Results are unique in trained artifacts,
    // so later duplicate rules can never match after the first rule fires.
    std::unordered_map<uint64_t, MergeInfo> merge_lookup_;
    bool fast_merge_order_ = true;
    std::unordered_map<std::string, int32_t> special_tokens_;
    std::unordered_map<int32_t, std::string> special_token_aliases_;

    std::vector<int32_t> encode_impl(const std::string& text) const;
    void append_regular(std::string_view text, std::vector<int32_t>& output, EncodingScratch& scratch) const;
    void append_word(std::string_view bytes, std::vector<int32_t>& output, EncodingScratch& scratch) const;
    void encode_word_into(std::string_view bytes, EncodingScratch& scratch) const;
    void invalidate_cache(); // Caller holds the exclusive model lock.

    bool load_vocab(const std::string& path);
    bool load_merges(const std::string& path);
};

} // namespace curriculum
