#include "tokenizer.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <exception>
#include <iterator>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace curriculum {
namespace {

bool is_hex(char c) {
    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string vocab_string_to_bytes(const std::string& display) {
    std::string result;
    result.reserve(display.size());

    size_t i = 0;
    while (i < display.size()) {
        if (i + 5 < display.size() &&
            display[i] == '<' && display[i + 1] == '0' &&
            display[i + 2] == 'x' && is_hex(display[i + 3]) &&
            is_hex(display[i + 4]) && display[i + 5] == '>') {
            const int value = hex_value(display[i + 3]) * 16 + hex_value(display[i + 4]);
            result.push_back(static_cast<char>(value));
            i += 6;
        } else {
            result.push_back(display[i]);
            ++i;
        }
    }
    return result;
}

std::string bytes_to_vocab_string(const std::string& bytes) {
    std::string result;
    result.reserve(bytes.size());
    for (unsigned char c : bytes) {
        if (c >= 0x20 && c <= 0x7E &&
            c != static_cast<unsigned char>('<')) {
            result.push_back(static_cast<char>(c));
        } else {
            char buffer[7];
            std::snprintf(buffer, sizeof(buffer), "<0x%02X>", c);
            result += buffer;
        }
    }
    return result;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool parse_hex4(const std::string& input, size_t pos, uint32_t& value) {
    if (pos + 4 > input.size()) return false;
    value = 0;
    for (size_t j = 0; j < 4; ++j) {
        const int h = hex_value(input[pos + j]);
        if (h < 0) return false;
        value = (value << 4) | static_cast<uint32_t>(h);
    }
    return true;
}

bool parse_json_string(
    const std::string& content,
    size_t& i,
    std::string& output
) {
    if (i >= content.size() || content[i] != '"') return false;
    ++i;
    output.clear();

    while (i < content.size()) {
        const char c = content[i++];

        if (c == '"') return true;

        if (c != '\\') {
            output.push_back(c);
            continue;
        }

        if (i >= content.size()) return false;

        const char esc = content[i++];

        switch (esc) {
            case '"':
                output.push_back('"');
                break;

            case '\\':
                output.push_back('\\');
                break;

            case '/':
                output.push_back('/');
                break;

            case 'b':
                output.push_back('\b');
                break;

            case 'f':
                output.push_back('\f');
                break;

            case 'n':
                output.push_back('\n');
                break;

            case 'r':
                output.push_back('\r');
                break;

            case 't':
                output.push_back('\t');
                break;

            case 'u': {
                uint32_t cp = 0;

                if (!parse_hex4(content, i, cp)) {
                    return false;
                }

                i += 4;

                // Handle a UTF-16 surrogate pair when present.
                if (cp >= 0xD800 && cp <= 0xDBFF &&
                    i + 6 <= content.size() &&
                    content[i] == '\\' &&
                    content[i + 1] == 'u') {

                    uint32_t low = 0;

                    if (parse_hex4(content, i + 2, low) &&
                        low >= 0xDC00 &&
                        low <= 0xDFFF) {

                        cp = 0x10000 +
                             ((cp - 0xD800) << 10) +
                             (low - 0xDC00);

                        i += 6;
                    }
                }

                append_utf8(output, cp);
                break;
            }

            default:
                return false;
        }
    }

    return false;
}

void skip_json_space(const std::string& content, size_t& i) {
    while (i < content.size() &&
           (content[i] == ' ' ||
            content[i] == '\n' ||
            content[i] == '\r' ||
            content[i] == '\t')) {
        ++i;
    }
}

bool parse_vocab_json(
    const std::string& content,
    std::unordered_map<std::string, int32_t>& output
) {
    size_t i = 0;

    skip_json_space(content, i);

    if (i >= content.size() || content[i++] != '{') {
        return false;
    }

    while (true) {
        skip_json_space(content, i);

        if (i < content.size() && content[i] == '}') {
            return true;
        }

        std::string key;

        if (!parse_json_string(content, i, key)) {
            return false;
        }

        skip_json_space(content, i);

        if (i >= content.size() || content[i++] != ':') {
            return false;
        }

        skip_json_space(content, i);

        if (i >= content.size() ||
            !std::isdigit(static_cast<unsigned char>(content[i]))) {
            return false;
        }

        int64_t value = 0;

        while (i < content.size() &&
               std::isdigit(static_cast<unsigned char>(content[i]))) {

            value = value * 10 + (content[i++] - '0');

            if (value > std::numeric_limits<int32_t>::max()) {
                return false;
            }
        }

        output.emplace(
            std::move(key),
            static_cast<int32_t>(value)
        );

        skip_json_space(content, i);

        if (i < content.size() && content[i] == ',') {
            ++i;
            continue;
        }

        if (i < content.size() && content[i] == '}') {
            return true;
        }

        return false;
    }
}

struct CodePoint {
    uint32_t value;
    size_t length;
};

CodePoint decode_utf8(std::string_view text, size_t i) {
    const auto first =
        static_cast<unsigned char>(text[i]);

    if (first < 0x80) {
        return {first, 1};
    }

    auto continuation = [&](size_t offset) {
        return i + offset < text.size() &&
               (static_cast<unsigned char>(text[i + offset]) & 0xC0) == 0x80;
    };

    if ((first & 0xE0) == 0xC0 &&
        continuation(1)) {

        return {
            static_cast<uint32_t>((first & 0x1F) << 6) |
            (static_cast<unsigned char>(text[i + 1]) & 0x3F),
            2
        };
    }

    if ((first & 0xF0) == 0xE0 &&
        continuation(1) &&
        continuation(2)) {

        return {
            static_cast<uint32_t>((first & 0x0F) << 12) |
            static_cast<uint32_t>(
                (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6
            ) |
            (static_cast<unsigned char>(text[i + 2]) & 0x3F),
            3
        };
    }

    if ((first & 0xF8) == 0xF0 &&
        continuation(1) &&
        continuation(2) &&
        continuation(3)) {

        return {
            static_cast<uint32_t>((first & 0x07) << 18) |
            static_cast<uint32_t>(
                (static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12
            ) |
            static_cast<uint32_t>(
                (static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6
            ) |
            (static_cast<unsigned char>(text[i + 3]) & 0x3F),
            4
        };
    }

    // Invalid UTF-8: preserve the byte as its own token.
    return {first, 1};
}

bool is_shared_whitespace(uint32_t cp) {
    return (cp >= 0x0009 && cp <= 0x000D) ||
           cp == 0x0020 ||
           cp == 0x0085 ||
           cp == 0x00A0 ||
           cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 ||
           cp == 0x2029 ||
           cp == 0x202F ||
           cp == 0x205F ||
           cp == 0x3000;
}

constexpr size_t kMaxNonWhitespaceBytes = 256;

uint64_t pair_key(int32_t left, int32_t right) {
    return (
        static_cast<uint64_t>(
            static_cast<uint32_t>(left)
        ) << 32
    ) |
    static_cast<uint32_t>(right);
}

constexpr uint32_t kNoNode =
    std::numeric_limits<uint32_t>::max();

struct MergeHeapEntry {
    size_t rank;
    uint32_t left_node;
    uint64_t generation;
};

struct MergeHeapEntryGreater {
    bool operator()(
        const MergeHeapEntry& a,
        const MergeHeapEntry& b
    ) const {
        if (a.rank != b.rank) {
            return a.rank > b.rank;
        }

        return a.left_node > b.left_node;
    }
};

} // namespace

struct Tokenizer::EncodingScratch {
    std::vector<int32_t> tokens, encoded, legacy_next;
    std::vector<uint32_t> previous, next;
    std::vector<uint64_t> generation;
    std::vector<MergeHeapEntry> heap;
};

bool Tokenizer::load(
    const std::string& vocab_path,
    const std::string& merges_path
) {
    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    invalidate_cache();
    id_to_token_.clear();
    token_to_id_.clear();
    merges_.clear();
    merge_lookup_.clear();
    fast_merge_order_ = true;
    special_tokens_.clear();
    special_token_aliases_.clear();

    if (!load_vocab(vocab_path)) {
        std::cerr
            << "Failed to load vocab from "
            << vocab_path
            << '\n';

        return false;
    }

    if (!load_merges(merges_path)) {
        std::cerr
            << "Failed to load merges from "
            << merges_path
            << '\n';

        return false;
    }

    return true;
}

bool Tokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        return false;
    }

    const std::string content(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    std::unordered_map<std::string, int32_t> raw_vocab;

    if (!parse_vocab_json(content, raw_vocab) ||
        raw_vocab.empty()) {
        return false;
    }

    int32_t max_id = -1;

    for (const auto& entry : raw_vocab) {
        max_id = std::max(
            max_id,
            entry.second
        );
    }

    id_to_token_.assign(
        static_cast<size_t>(max_id) + 1,
        std::string{}
    );

    std::vector<bool> seen_ids(
        id_to_token_.size(),
        false
    );

    for (const auto& [display, id] : raw_vocab) {
        if (id < 0 ||
            static_cast<size_t>(id) >= id_to_token_.size()) {
            return false;
        }

        const std::string bytes =
            vocab_string_to_bytes(display);

        if (bytes.empty()) {
            std::cerr
                << "Empty token in vocabulary at ID "
                << id
                << '\n';

            return false;
        }

        if (seen_ids[static_cast<size_t>(id)]) {
            std::cerr
                << "Duplicate vocabulary ID "
                << id
                << '\n';

            return false;
        }

        if (token_to_id_.find(bytes) !=
            token_to_id_.end()) {

            std::cerr
                << "Duplicate vocabulary token bytes at ID "
                << id
                << '\n';

            return false;
        }

        seen_ids[static_cast<size_t>(id)] = true;

        id_to_token_[id] = bytes;
        token_to_id_[bytes] = id;
    }

    for (size_t id = 0; id < seen_ids.size(); ++id) {
        if (!seen_ids[id]) {
            std::cerr
                << "Missing vocabulary ID "
                << id
                << '\n';

            return false;
        }
    }

    for (int32_t id = 0; id < 256; ++id) {
        if (static_cast<size_t>(id) >=
            id_to_token_.size()) {
            return false;
        }

        const std::string expected(
            1,
            static_cast<char>(id)
        );

        if (id_to_token_[id] != expected) {
            std::cerr
                << "Invalid base-byte token at ID "
                << id
                << '\n';

            return false;
        }
    }

    return true;
}

bool Tokenizer::load_merges(
    const std::string& path
) {
    std::ifstream file(path);

    if (!file) {
        return false;
    }

    merge_lookup_.clear();
    fast_merge_order_ = true;

    int32_t previous_result = 255;

    std::string line;
    size_t line_number = 0;

    while (std::getline(file, line)) {
        ++line_number;

        if (line.empty() ||
            line[0] == '#') {
            continue;
        }

        int64_t left = -1;
        int64_t right = -1;
        int64_t result = -1;

        std::istringstream stream(line);
        std::string extra;

        if (!(stream >> left >> right >> result) ||
            (stream >> extra)) {

            std::cerr
                << "Invalid merge rule at line "
                << line_number
                << '\n';

            return false;
        }

        const auto valid_id = [&](int64_t id) {
            return id >= 0 &&
                   static_cast<size_t>(id) <
                       id_to_token_.size();
        };

        if (!valid_id(left) ||
            !valid_id(right) ||
            !valid_id(result)) {

            std::cerr
                << "Out-of-range merge ID at line "
                << line_number
                << '\n';

            return false;
        }

        const std::string expected =
            id_to_token_[left] +
            id_to_token_[right];

        if (id_to_token_[result] != expected) {
            std::cerr
                << "Merge result bytes do not match at line "
                << line_number
                << '\n';

            return false;
        }

        const int32_t left_id =
            static_cast<int32_t>(left);

        const int32_t right_id =
            static_cast<int32_t>(right);

        const int32_t result_id =
            static_cast<int32_t>(result);

        const size_t rank =
            merges_.size();

        // A trained vocabulary assigns monotonically increasing IDs to merge
        // results. That makes newly-created pairs strictly later rules, which
        // is the condition needed by the indexed heap encoder below. Keep the
        // legacy path for unusual hand-authored files.
        if (result_id <= previous_result ||
            result_id <= left_id ||
            result_id <= right_id) {

            fast_merge_order_ = false;
        }

        previous_result = result_id;

        const auto [_, inserted] =
            merge_lookup_.emplace(
                pair_key(left_id, right_id),
                MergeInfo{rank, result_id}
            );

        if (!inserted) {
            fast_merge_order_ = false;
        }

        merges_.emplace_back(
            left_id,
            right_id,
            result_id
        );
    }

    return true;
}

void Tokenizer::encode_word_into(std::string_view bytes, EncodingScratch& scratch) const {
    auto& tokens = scratch.tokens;
    auto& encoded = scratch.encoded;
    tokens.resize(bytes.size());
    // char may be signed; byte IDs must remain in [0, 255].
    for (size_t i = 0; i < bytes.size(); ++i) tokens[i] = static_cast<unsigned char>(bytes[i]);
    encoded.clear();
    if (tokens.size() < 2 || merges_.empty()) {
        encoded.assign(tokens.begin(), tokens.end());
        return;
    }
    if (!fast_merge_order_) {
        auto& next = scratch.legacy_next;
        for (const auto& [left, right, result] : merges_) {
            next.clear();
            for (size_t i = 0; i < tokens.size();) {
                if (i + 1 < tokens.size() && tokens[i] == left && tokens[i + 1] == right) {
                    next.push_back(result);
                    i += 2;
                } else next.push_back(tokens[i++]);
            }
            tokens.swap(next);
        }
        encoded.assign(tokens.begin(), tokens.end());
        return;
    }
    if (tokens.size() == 2) {
        const auto found = merge_lookup_.find(pair_key(tokens[0], tokens[1]));
        if (found != merge_lookup_.end()) encoded.push_back(found->second.result);
        else encoded.assign(tokens.begin(), tokens.end());
        return;
    }
    if (tokens.size() >= kNoNode) throw std::length_error("pretoken piece exceeds node index range");
    const size_t count = tokens.size();
    auto& previous = scratch.previous;
    auto& next = scratch.next;
    auto& generation = scratch.generation;
    auto& heap = scratch.heap;
    previous.assign(count, kNoNode);
    next.assign(count, kNoNode);
    generation.assign(count, 0);
    heap.clear();
    for (uint32_t i = 0; i + 1 < count; ++i) {
        next[i] = i + 1;
        previous[i + 1] = i;
    }
    const size_t no_rank = std::numeric_limits<size_t>::max();
    bool heap_ready = false;
    auto enqueue = [&](uint32_t left, size_t completed_rank) {
        if (left == kNoNode || next[left] == kNoNode) return;
        const auto found = merge_lookup_.find(pair_key(tokens[left], tokens[next[left]]));
        if (found == merge_lookup_.end() ||
            (completed_rank != no_rank && found->second.rank <= completed_rank)) return;
        heap.push_back({found->second.rank, left, generation[left]});
        if (heap_ready) std::push_heap(heap.begin(), heap.end(), MergeHeapEntryGreater{});
    };
    for (uint32_t i = 0; i + 1 < count; ++i) enqueue(i, no_rank);
    std::make_heap(heap.begin(), heap.end(), MergeHeapEntryGreater{});
    heap_ready = true;
    auto pop = [&]() {
        std::pop_heap(heap.begin(), heap.end(), MergeHeapEntryGreater{});
        const auto entry = heap.back();
        heap.pop_back();
        return entry;
    };
    size_t completed_rank = no_rank;
    while (!heap.empty()) {
        const auto first = pop();
        if (completed_rank != no_rank && first.rank <= completed_rank) continue;
        const size_t rank = first.rank;
        completed_rank = rank;
        auto merge_if_current = [&](const MergeHeapEntry& entry) {
            if (entry.left_node >= count || generation[entry.left_node] != entry.generation) return;
            const uint32_t left = entry.left_node;
            const uint32_t right = next[left];
            if (right == kNoNode) return;
            const auto found = merge_lookup_.find(pair_key(tokens[left], tokens[right]));
            if (found == merge_lookup_.end() || found->second.rank != rank) return;
            const uint32_t before = previous[left];
            const uint32_t after = next[right];
            tokens[left] = found->second.result;
            next[left] = after;
            if (after != kNoNode) previous[after] = left;
            next[right] = previous[right] = kNoNode;
            ++generation[left];
            ++generation[right];
            if (before != kNoNode) ++generation[before];
            enqueue(before, rank);
            enqueue(left, rank);
        };
        merge_if_current(first);
        while (!heap.empty() && heap.front().rank == rank) merge_if_current(pop());
    }
    for (uint32_t node = 0; node != kNoNode; node = next[node]) encoded.push_back(tokens[node]);
}

void Tokenizer::append_word(std::string_view bytes, std::vector<int32_t>& output,
                            EncodingScratch& scratch) const {
    if (bytes.size() == 1) {
        output.push_back(static_cast<unsigned char>(bytes[0]));
        return;
    }
    if (cache_capacity_ == 0 || bytes.size() > kMaxCachedPieceBytes) {
        encode_word_into(bytes, scratch);
        output.insert(output.end(), scratch.encoded.begin(), scratch.encoded.end());
        return;
    }
    const size_t shard_id = std::hash<std::string_view>{}(bytes) % kCacheShards;
    const size_t capacity = cache_capacity_ / kCacheShards + (shard_id < cache_capacity_ % kCacheShards);
    const bool cacheable = capacity != 0;
    auto& shard = piece_cache_[shard_id];
    if (cacheable) {
        std::lock_guard<std::mutex> lock(shard.mutex);
        const auto found = shard.index.find(bytes);
        if (found != shard.index.end()) {
            ++shard.hits;
            shard.lru.splice(shard.lru.begin(), shard.lru, found->second);
            const auto& ids = found->second->ids;
            output.insert(output.end(), ids.begin(), ids.end());
            return;
        }
        ++shard.misses;
    }
    encode_word_into(bytes, scratch);
    const auto& ids = scratch.encoded;
    if (cacheable) {
        std::lock_guard<std::mutex> lock(shard.mutex);
        // Another encoding thread can have filled this exact piece meanwhile.
        if (shard.index.find(bytes) == shard.index.end()) {
            shard.lru.push_front({std::string(bytes), ids});
            try {
                shard.index.emplace(std::string_view(shard.lru.front().bytes), shard.lru.begin());
            } catch (...) {
                shard.lru.pop_front();
                throw;
            }
            shard.payload_bytes += bytes.size() + ids.size() * sizeof(int32_t);
            while (shard.lru.size() > capacity || shard.payload_bytes > kCachePayloadBytes / kCacheShards) {
                const auto last = std::prev(shard.lru.end());
                shard.payload_bytes -= last->bytes.size() + last->ids.size() * sizeof(int32_t);
                shard.index.erase(std::string_view(last->bytes));
                shard.lru.erase(last);
            }
        }
    }
    output.insert(output.end(), ids.begin(), ids.end());
}

void Tokenizer::append_regular(std::string_view text, std::vector<int32_t>& output,
                               EncodingScratch& scratch) const {
    size_t i = 0;
    while (i < text.size()) {
        const size_t start = i;
        while (i < text.size()) {
            const auto cp = decode_utf8(text, i);
            if (!is_shared_whitespace(cp.value)) break;
            i += cp.length;
        }
        const size_t field_start = i;
        while (i < text.size()) {
            const auto cp = decode_utf8(text, i);
            if (is_shared_whitespace(cp.value) ||
                (i > field_start && i + cp.length - field_start > kMaxNonWhitespaceBytes)) break;
            i += cp.length;
        }
        append_word(text.substr(start, i - start), output, scratch);
    }
}

std::vector<int32_t> Tokenizer::encode_impl(const std::string& text) const {
    // Scratch holds only temporary arrays; cached results belong to the model.
    thread_local EncodingScratch scratch;
    std::vector<int32_t> output;
    output.reserve(text.size() / 3 + 8);
    const std::string_view view(text);
    if (special_tokens_.empty()) append_regular(view, output, scratch);
    else {
        size_t cursor = 0;
        while (cursor < text.size()) {
            size_t position = std::string::npos, length = 0;
            int32_t id = -1;
            for (const auto& special : special_tokens_) {
                const size_t found = text.find(special.first, cursor);
                if (found < position || (found != std::string::npos && found == position && special.first.size() > length)) {
                    position = found;
                    length = special.first.size();
                    id = special.second;
                }
            }
            if (position == std::string::npos) {
                append_regular(view.substr(cursor), output, scratch);
                break;
            }
            append_regular(view.substr(cursor, position - cursor), output, scratch);
            output.push_back(id);
            cursor = position + length;
        }
    }
    // A giant whitespace run must not permanently pin a giant per-thread arena.
    if (scratch.tokens.capacity() > 4096 || scratch.encoded.capacity() > 4096 ||
        scratch.legacy_next.capacity() > 4096 || scratch.heap.capacity() > 16384) scratch = EncodingScratch{};
    return output;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    return encode_impl(text);
}

void Tokenizer::invalidate_cache() {
    for (auto& shard : piece_cache_) {
        std::lock_guard<std::mutex> lock(shard.mutex);
        shard.index.clear();
        shard.lru.clear();
        shard.payload_bytes = shard.hits = shard.misses = 0;
    }
}

void Tokenizer::set_cache_capacity(size_t capacity) {
    if (capacity > 1048576) throw std::invalid_argument("cache capacity exceeds 1048576 pieces");
    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    invalidate_cache();
    cache_capacity_ = capacity;
}

void Tokenizer::clear_cache() {
    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    invalidate_cache();
}

std::unordered_map<std::string, size_t> Tokenizer::cache_info() const {
    std::shared_lock<std::shared_mutex> model_lock(model_mutex_);
    std::unordered_map<std::string, size_t> result{
        {"capacity", cache_capacity_}, {"entries", 0}, {"hits", 0}, {"misses", 0},
        {"payload_bytes", 0}, {"max_payload_bytes", kCachePayloadBytes}};
    for (auto& shard : piece_cache_) {
        std::lock_guard<std::mutex> lock(shard.mutex);
        result["entries"] += shard.index.size();
        result["hits"] += shard.hits;
        result["misses"] += shard.misses;
        result["payload_bytes"] += shard.payload_bytes;
    }
    return result;
}

size_t Tokenizer::vocab_size() const {
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    return id_to_token_.size();
}

std::string Tokenizer::decode(
    const std::vector<int32_t>& ids
) const {
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    std::string result;

    for (int32_t id : ids) {
        if (id < 0 ||
            static_cast<size_t>(id) >=
                id_to_token_.size()) {

            throw std::out_of_range(
                "token ID out of vocabulary range"
            );
        }

        const auto alias = special_token_aliases_.find(id);
        result += alias == special_token_aliases_.end() ? id_to_token_[id] : alias->second;
    }

    return result;
}

std::vector<std::vector<int32_t>>
Tokenizer::encode_batch(
    const std::vector<std::string>& texts, size_t num_threads
) const {
    if (num_threads > 64) throw std::invalid_argument("num_threads must be between 0 and 64");
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    std::vector<std::vector<int32_t>> result(texts.size());
    size_t workers = num_threads;
    if (workers == 0) {
        size_t bytes = 0;
        for (const auto& text : texts) bytes += text.size();
        workers = bytes >= 256 * 1024 ? std::min<size_t>(4, std::max(1u, std::thread::hardware_concurrency())) : 1;
    }
    workers = std::min(workers, texts.size());
    if (workers <= 1) {
        for (size_t i = 0; i < texts.size(); ++i) result[i] = encode_impl(texts[i]);
        return result;
    }
    std::atomic<size_t> cursor{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto work = [&]() {
        try {
            while (!failed.load(std::memory_order_relaxed)) {
                const size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
                if (i >= texts.size()) break;
                result[i] = encode_impl(texts[i]);
            }
        } catch (...) {
            std::lock_guard<std::mutex> error_lock(error_mutex);
            if (!error) error = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    std::vector<std::thread> threads;
    try {
        for (size_t i = 1; i < workers; ++i) threads.emplace_back(work);
    } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        for (auto& thread : threads) thread.join();
        throw;
    }
    work();
    for (auto& thread : threads) thread.join();
    if (error) std::rethrow_exception(error);
    return result;
}

std::string Tokenizer::id_to_token(
    int32_t id
) const {
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    if (id < 0 ||
        static_cast<size_t>(id) >=
            id_to_token_.size()) {
        return {};
    }

    const auto alias = special_token_aliases_.find(id);
    return bytes_to_vocab_string(alias == special_token_aliases_.end() ? id_to_token_[id] : alias->second);
}

int32_t Tokenizer::token_to_id(
    const std::string& token
) const {
    std::shared_lock<std::shared_mutex> lock(model_mutex_);
    const auto it =
        token_to_id_.find(token);

    return it == token_to_id_.end()
        ? -1
        : it->second;
}

void Tokenizer::set_special_tokens(
    const std::vector<std::string>& tokens
) {
    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    std::unordered_map<std::string, int32_t> configured;
    for (const auto& token : tokens) {
        const auto found = token_to_id_.find(token);
        if (token.empty() || found == token_to_id_.end() || found->second < 256) {
            throw std::invalid_argument("special token is missing from the loaded vocabulary: " + token);
        }
        if (!configured.emplace(token, found->second).second) {
            throw std::invalid_argument("duplicate special token: " + token);
        }
        for (const auto& [left, right, result] : merges_) {
            if (left == found->second || right == found->second || result == found->second) {
                throw std::invalid_argument("special token ID participates in an ordinary merge: " + token);
            }
        }
    }
    invalidate_cache();
    special_tokens_ = std::move(configured);
    special_token_aliases_.clear();
}

void Tokenizer::configure_special_tokens(
    const std::unordered_map<std::string, int32_t>& token_ids,
    const std::unordered_map<int32_t, std::string>& aliases,
    size_t expected_vocab_size
) {
    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    // Legacy artifacts may keep tail controls in metadata rather than vocab.json.
    // Validate every addition before committing; existing ordinary IDs are immutable.
    std::unordered_map<int32_t, std::string> additions;
    std::unordered_set<int32_t> protected_ids;
    size_t final_size = id_to_token_.size();
    for (const auto& [token, id] : token_ids) {
        if (token.empty() || id < 256 || !protected_ids.insert(id).second) {
            throw std::invalid_argument("invalid or duplicate canonical special-token ID");
        }
        const auto existing = token_to_id_.find(token);
        if (existing != token_to_id_.end() && existing->second != id) {
            throw std::invalid_argument("configured special-token ID disagrees with vocabulary");
        }
        if (static_cast<size_t>(id) < id_to_token_.size()) {
            if (id_to_token_[id] != token) {
                throw std::invalid_argument("special token would overwrite an existing vocabulary ID");
            }
        } else {
            additions.emplace(id, token);
        }
        final_size = std::max(final_size, static_cast<size_t>(id) + 1);
    }
    for (const auto& [id, token] : aliases) {
        if (id < 256 || token_ids.find(token) == token_ids.end() ||
            !protected_ids.insert(id).second) {
            throw std::invalid_argument("invalid or conflicting decode-only alias");
        }
        if (static_cast<size_t>(id) >= id_to_token_.size()) {
            additions.emplace(id, token);
        }
        final_size = std::max(final_size, static_cast<size_t>(id) + 1);
    }
    if (final_size != expected_vocab_size ||
        final_size - id_to_token_.size() != additions.size()) {
        throw std::invalid_argument("configured vocabulary size disagrees with vocabulary and explicit controls");
    }
    for (const auto& [left, right, result] : merges_) {
        if (protected_ids.count(left) || protected_ids.count(right) || protected_ids.count(result)) {
            throw std::invalid_argument("configured control or alias ID participates in an ordinary merge");
        }
    }
    // Unique additions with the correct span count are necessarily contiguous.
    id_to_token_.reserve(final_size);
    for (size_t id = id_to_token_.size(); id < final_size; ++id) {
        id_to_token_.push_back(additions.at(static_cast<int32_t>(id)));
    }
    for (const auto& [token, id] : token_ids) {
        token_to_id_.emplace(token, id);
    }
    invalidate_cache();
    special_tokens_ = token_ids;
    special_token_aliases_ = aliases;
}

void Tokenizer::set_special_token_aliases(
    const std::unordered_map<int32_t, std::string>& aliases
) {
    std::unique_lock<std::shared_mutex> lock(model_mutex_);
    for (const auto& [id, token] : aliases) {
        if (id < 256 || static_cast<size_t>(id) >= id_to_token_.size() ||
            special_tokens_.find(token) == special_tokens_.end()) {
            throw std::invalid_argument("invalid decode-only special-token alias");
        }
        for (const auto& special : special_tokens_) {
            if (special.second == id) {
                throw std::invalid_argument("decode-only alias conflicts with a canonical special ID");
            }
        }
        for (const auto& [left, right, result] : merges_) {
            if (left == id || right == id || result == id) {
                throw std::invalid_argument("decode-only alias would replace a learned token ID");
            }
        }
    }
    invalidate_cache();
    special_token_aliases_ = aliases;
}

} // namespace curriculum
