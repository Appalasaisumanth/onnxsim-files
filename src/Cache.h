#pragma once
#include "Common.h"
#include <array>
#include <deque>
#include <vector>
#include <cstdint>

// ============================================================
// Geometry
// ============================================================
static constexpr uint32_t BLOCK_SIZE       = 128;
static constexpr uint32_t NUM_WAYS         = 16;
static constexpr uint32_t CACHE_SIZE_BYTES = 64u * 1024u * 1024u; // 64 MB

static constexpr uint32_t NUM_BLOCKS = CACHE_SIZE_BYTES / BLOCK_SIZE; // 524288
static constexpr uint32_t NUM_SETS   = NUM_BLOCKS / NUM_WAYS;         // 32768
static constexpr uint32_t BLOCK_BITS = 7;
static constexpr uint32_t SET_BITS   = 15;
static constexpr uint64_t SET_MASK   = (1u << SET_BITS) - 1;

// ============================================================
// Tuning constants
// ============================================================

// How many 128B blocks we issue to HBM in one batch per pending object.
// Real hardware issues as many as the HBM controller can accept.
// Setting this high means fewer cycle() iterations needed to drain
// a large object — critical for OPT/Mamba 786KB weight matrices.
// 786KB / 128B = 6144 blocks. We issue all at once in one batch.
static constexpr uint32_t MAX_MISS_Q_DEPTH  = 16834;  // large enough for one 8MB object
static constexpr uint32_t MAX_PENDING_OBJS  = 2048;   // max simultaneous fetches
static constexpr uint32_t MAX_IN_PER_CYCLE  = 16;
// Replace the broken duplicate definition and set correct size
static constexpr uint32_t BLOCK_POOL_SIZE = MAX_MISS_Q_DEPTH * 4
                                          + MAX_PENDING_OBJS * 32
                                          + 8192;  // ~150K — pool should never fall back

static constexpr uint32_t WB_POOL_SIZE = MAX_PENDING_OBJS * 2 + 4096; // dedicated wb pool
// ============================================================
// Large-object burst transfer model
// ============================================================
// Objects above this threshold are sent as a single HBM burst.
// One MemoryAccess issued, cycle-accurate latency computed,
// no per-block tracking — matches real burst controller behavior.
static constexpr uint32_t LARGE_OBJ_THRESHOLD   = 384u * 1024u; // 384 KB

// HBM2 peak bandwidth: 256 GB/s across all channels.
// With 16 banks each on one channel: 256/16 = 16 GB/s per bank.
// At your sim frequency (assume 1GHz → 1 cycle = 1ns):
//   16 GB/s = 16 bytes/ns = 16 bytes/cycle per bank.
// Tune HBM_BYTES_PER_CYCLE to match your SimulationConfig freq.
static constexpr uint32_t HBM_BYTES_PER_CYCLE   = 16;  // bytes per cycle per bank — TUNE THIS

// Minimum latency floor for any HBM access (row activation + CAS)
// HBM2 typical: ~80ns. At 1GHz = 80 cycles.
static constexpr uint32_t HBM_MIN_LATENCY_CYCLES = 80;
// ============================================================
// CacheEntry — 8 bytes
//   [41:0]  tag     (42 bits)
//   [42]    valid
//   [43]    dirty
//   [47:44] _pad
//   [63:48] core_id (0xFFFF = none)
// ============================================================
struct CacheEntry {
    uint64_t tag     : 42;
    uint64_t valid   :  1;
    uint64_t dirty   :  1;
    uint64_t _pad    :  4;
    uint64_t core_id : 16;

    void reset() {
        tag = 0; valid = 0; dirty = 0; _pad = 0; core_id = 0xFFFF;
    }
    void    set_core(int32_t c) { core_id = (uint16_t)(int16_t)c; }
    int32_t get_core()    const { return (int32_t)(int16_t)(uint16_t)core_id; }
};
static_assert(sizeof(CacheEntry) == 8, "CacheEntry must be 8 bytes");

// ============================================================
// CacheSet — 16 ways + PLRU + valid bitmask
// ============================================================
struct CacheSet {
    std::array<CacheEntry, NUM_WAYS> ways;
    uint16_t plru       = 0;
    uint16_t valid_mask = 0;

    uint32_t free_way() const {
        if (valid_mask == 0xFFFFu) return NUM_WAYS;
        return static_cast<uint32_t>(__builtin_ctz(~(uint32_t)valid_mask));
    }

    uint32_t find_tag(uint64_t tag) const {
        uint16_t mask = valid_mask;
        while (mask) {
            uint32_t w = static_cast<uint32_t>(__builtin_ctz(mask));
            if (ways[w].tag == tag) return w;
            mask &= static_cast<uint16_t>(mask - 1u);
        }
        return NUM_WAYS;
    }

    uint32_t lru_way() const {
        uint32_t node = 0;
        for (int d = 0; d < 4; ++d) {
            bool go_right = (plru >> node) & 1u;
            node = go_right ? (2u * node + 2u) : (2u * node + 1u);
        }
        return node - (NUM_WAYS - 1u);
    }

    void touch(uint32_t way) {
        uint32_t node = 0;
        for (int d = 0; d < 4; ++d) {
            uint32_t right = 2u * node + 2u;
            uint32_t rl    = right;
            for (int d2 = d + 1; d2 < 4; ++d2) rl = 2u * rl + 1u;
            bool go_right = (way >= rl - (NUM_WAYS - 1u));
            if (go_right)
                plru |=  static_cast<uint16_t>(1u << node);
            else
                plru &= static_cast<uint16_t>(~(1u << node));
            node = go_right ? right : (2u * node + 1u);
        }
    }

    void set_valid  (uint32_t w) { valid_mask |=  static_cast<uint16_t>(1u << w); }
    void clear_valid(uint32_t w) { valid_mask &= static_cast<uint16_t>(~(1u << w)); }
};

// ============================================================
// PendingObject
//
// Tracks one in-flight multi-block fetch.
//
// KEY DESIGN: instead of issuing one MemoryAccess per 128B block
// (which creates millions of objects for large tensors), we issue
// ONE MemoryAccess per object that carries the full object size.
// The HBM controller models bandwidth correctly using object_size.
// We track how many blocks are expected and count responses.
//
// This matches real hardware behavior:
//   - Cache controller issues a burst request for N lines
//   - HBM returns them pipelined
//   - Cache fills each line as it arrives
//   - Core is released when last line arrives
// ============================================================
struct PendingObject {
    uint32_t  total_blocks    = 0;   // ceil(obj_size / BLOCK_SIZE)
    uint32_t  issued_blocks   = 0;   // blocks sent to HBM so far
    uint32_t  returned_blocks = 0;   // blocks received from HBM
    addr_type base_addr       = 0;
    uint32_t  obj_size        = 0;
    uint32_t  original_id     = 0;   // req->id from simulator
    bool      is_write        = false;
    bool      active          = false;
    int32_t   core_id         = -1;
    std::vector<MemoryAccess*> waiting_reqs;
    uint32_t  active_slots_idx = 0;  // ADD: index of this slot in _active_slots for O(1) remove
    uint32_t  burst_countdown   = 0;   // ADD: cycles remaining for large-obj burst
    bool      is_burst_mode     = false; // ADD: true = single-burst, no block tracking

    void reset() {
        total_blocks = issued_blocks = returned_blocks = 0;
        base_addr = 0; obj_size = 0; original_id = 0;
        is_write = false; active = false; core_id = -1;
        burst_countdown = 0; is_burst_mode = false;  // ADD
        waiting_reqs.clear();
    
    }
};

// ============================================================
// BlockPool — reusable MemoryAccess pool
// Sized to hold all in-flight block requests without heap alloc
// ============================================================
struct BlockPool {
    std::vector<MemoryAccess> pool;
    std::vector<uint32_t>     free_list;

    explicit BlockPool(uint32_t size) : pool(size) {
        free_list.reserve(size);
        for (uint32_t i = 0; i < size; ++i)
            free_list.push_back(i);
    }

    MemoryAccess* alloc() {
        if (!free_list.empty()) {
            uint32_t idx = free_list.back();
            free_list.pop_back();
            return &pool[idx];
        }
        return new MemoryAccess(); // fallback — logged as pool exhaustion
    }

    void release(MemoryAccess* p) {
        ptrdiff_t idx = p - pool.data();
        if (idx >= 0 && static_cast<size_t>(idx) < pool.size())
            free_list.push_back(static_cast<uint32_t>(idx));
        else
            delete p;
    }

    bool is_pool_ptr(const MemoryAccess* p) const {
        ptrdiff_t idx = p - pool.data();
        return idx >= 0 && static_cast<size_t>(idx) < pool.size();
    }
};

// ============================================================
// Cache
// ============================================================
class Cache {
public:
    Cache(uint32_t bank_id, SimulationConfig config);

    void push_request(MemoryAccess* req);

    MemoryAccess* top_miss_request();
    void          pop_miss_request();

    bool          has_wb_request();
    MemoryAccess* top_wb_request();
    void          pop_wb_request();

    void          push_memory_response(MemoryAccess* resp);

    MemoryAccess* top_response();
    void          pop_response();

    void cycle();
    void print_stats() const;

private:
    static uint32_t  set_index (addr_type a) {
        return static_cast<uint32_t>((a >> BLOCK_BITS) & SET_MASK);
    }
    static uint64_t  tag_bits  (addr_type a) {
        return a >> (BLOCK_BITS + SET_BITS);
    }
    static addr_type block_base(addr_type a) {
        return a & ~static_cast<addr_type>(BLOCK_SIZE - 1);
    }

    bool check_hit   (addr_type addr, uint32_t& si, uint32_t& way);
    void fill        (addr_type baddr, bool is_write, int32_t core_id);
    void writeback   (uint32_t si, uint32_t w);
    static uint32_t compute_burst_cycles(uint32_t obj_size_bytes);

    // Issue next batch of block requests for a pending object.
    // Sends up to MAX_MISS_Q_DEPTH - current_depth blocks in one call.
    void issue_blocks(uint32_t slot, PendingObject& po);

    uint32_t find_pending_slot (uint32_t obj_id) const;
    uint32_t alloc_pending_slot();
    void     free_pending_slot (uint32_t slot);

    bool process_one_request();
    

private:
    uint32_t   _bank_id;
    uint64_t   _size;
    uint32_t   _data_width;
    cycle_type _local_cycle = 0;


    std::array<CacheSet, NUM_SETS> _sets;

    std::array<PendingObject, MAX_PENDING_OBJS> _pending;
    std::vector<uint32_t> _active_slots;  // indices of active pending objects
    uint32_t _active_pending = 0;

    std::deque<MemoryAccess*> _in_q;
    std::deque<MemoryAccess*> _miss_q;
    std::deque<MemoryAccess*> _wb_q;
    std::deque<MemoryAccess*> _out_q;
    BlockPool _pool;
BlockPool _wb_pool;                          // ADD: separate pool for writebacks

std::vector<uint32_t> _free_slots;           // ADD: O(1) alloc/free for pending slots
std::unordered_map<uint32_t, uint32_t> _id_to_slot; // ADD: O(1) find_pending_slot

    // Stats
    uint64_t _hits         = 0;
    uint64_t _misses       = 0;
    uint64_t _read_hits    = 0;
    uint64_t _write_hits   = 0;
    uint64_t _evictions    = 0;
    uint64_t _writebacks   = 0;
    uint64_t _conflicts    = 0;
    uint64_t _pool_fallback= 0;  // times pool was exhausted — should be 0
};