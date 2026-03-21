#pragma once
#include "Common.h"
#include <array>
#include <deque>
#include <unordered_map>
#include <vector>
#include <cstdint>

// ============================================================
// Geometry — all derived from these three constants
// ============================================================
static constexpr uint32_t BLOCK_SIZE       = 256;          // bytes per cache line
static constexpr uint32_t NUM_WAYS         = 8;            // set associativity
static constexpr uint32_t CACHE_SIZE_BYTES = 4u*1024u*1024u; // 4 MB

static constexpr uint32_t NUM_BLOCKS       = CACHE_SIZE_BYTES / BLOCK_SIZE;       // 16 384
static constexpr uint32_t NUM_SETS         = NUM_BLOCKS / NUM_WAYS;               // 2 048
static constexpr uint32_t BLOCK_BITS       = 8;   // log2(256)   — byte offset
static constexpr uint32_t SET_BITS         = 11;  // log2(2048)  — set index
static constexpr uint64_t SET_MASK         = (1u << SET_BITS) - 1;
// tag  = addr >> (BLOCK_BITS + SET_BITS)   [45 bits, stored as uint64_t]

static constexpr uint32_t MAX_MISS_Q_DEPTH = 256;

// ============================================================
// CacheEntry — 8 bytes per way, single uint64_t bitfield
//
// Bit layout (LSB → MSB):
//   [44: 0]  tag      (45 bits) — addr >> (BLOCK_BITS + SET_BITS)
//   [45]     valid    ( 1 bit)
//   [46]     dirty    ( 1 bit)
//   [47]     _pad     ( 1 bit)
//   [63:48]  core_id  (16 bits, stored as uint16_t, 0xFFFF = -1/none)
//
// Everything lives in one machine word → no struct padding ever.
// 16 384 entries × 8 B = 128 KB total array.
// ============================================================
struct CacheEntry {
  uint64_t tag     : 45;
  uint64_t valid   :  1;
  uint64_t dirty   :  1;
  uint64_t _pad    :  1;
  uint64_t core_id : 16;   // stored unsigned; 0xFFFF means "no core"

  void reset() {
    tag     = 0;
    valid   = 0;
    dirty   = 0;
    _pad    = 0;
    core_id = 0xFFFF;
  }

  // Helpers to set/get core_id as a signed int16_t
  void     set_core(int32_t c) { core_id = (uint16_t)(int16_t)c; }
  int32_t  get_core() const    { return (int32_t)(int16_t)(uint16_t)core_id; }
};
static_assert(sizeof(CacheEntry) == 8, "CacheEntry must be 8 bytes");

// ============================================================
// CacheSet — fixed-size array of ways + 1-byte PLRU tree
//
// 8-way pseudo-LRU (binary tree, 7 internal nodes in uint8_t):
//
//          [0]
//         /   \
//       [1]   [2]
//       / \   / \
//     [3][4][5][6]
//     /\ /\ /\ /\
//    w0 w1 w2 w3 w4 w5 w6 w7
//
// Each internal node bit points toward the "LRU subtree":
//   0 = LRU is in LEFT child,  1 = LRU is in RIGHT child.
// On access(way): flip bits on the root→leaf path (3 bit-flips).
// On evict():     follow bits down the tree (3 comparisons) → O(1).
// ============================================================
struct CacheSet {
  std::array<CacheEntry, NUM_WAYS> ways;
  uint8_t plru = 0;   // 7 internal-node bits packed into one byte

  // Return the way index of the pseudo-LRU victim (O(log W))
  uint32_t lru_way() const {
    uint32_t node = 0;
    for (int depth = 0; depth < 3; ++depth) {
      bool go_right = (plru >> node) & 1;
      node = go_right ? (2*node + 2) : (2*node + 1);
    }
    // node is now a leaf index 1..7; map to way 0..7
    return node - (NUM_WAYS - 1);   // leaf offset within bottom level
  }

  // Update PLRU after accessing 'way' (flip 3 bits toward root)
  void touch(uint32_t way) {
    // Walk root → leaf, flipping each node to point AWAY from this way
    uint32_t node = 0;
    uint32_t leaf = way + (NUM_WAYS - 1);   // leaf node index
    for (int depth = 0; depth < 3; ++depth) {
      uint32_t left  = 2*node + 1;
      uint32_t right = 2*node + 2;
      // Does the path to our leaf go right at this node?
      bool go_right = (leaf >= right + (right > left ? 0 : 0));
      // Simpler: check if leaf is in the right subtree of node
      // Right subtree of node n covers leaves: (2n+2)'s subtree
      // Threshold: first leaf of right subtree
      uint32_t right_leaf_start = right;
      for (int d2 = depth+1; d2 < 3; ++d2) right_leaf_start = 2*right_leaf_start + 1;
      go_right = (way >= right_leaf_start - (NUM_WAYS - 1));

      // Flip this node to point AWAY from the accessed way
      if (go_right)
        plru |=  (1u << node);   // set bit → left is LRU
      else
        plru &= ~(1u << node);   // clear bit → right is LRU

      node = go_right ? right : left;
    }
  }

  // Find a way whose tag matches (returns NUM_WAYS if not found)
  uint32_t find_tag(uint64_t tag) const {
    for (uint32_t w = 0; w < NUM_WAYS; ++w)
      if (ways[w].valid && ways[w].tag == tag) return w;
    return NUM_WAYS;   // miss
  }

  // Find an invalid (empty) way, or NUM_WAYS if all are valid
  uint32_t free_way() const {
    for (uint32_t w = 0; w < NUM_WAYS; ++w)
      if (!ways[w].valid) return w;
    return NUM_WAYS;
  }
};

// ============================================================
// PendingObject — in-flight multi-block fetch
// ============================================================
struct PendingObject {
  uint32_t total_blocks    = 0;
  uint32_t issued_blocks   = 0;
  uint32_t returned_blocks = 0;
  addr_type base_addr      = 0;
  uint32_t  obj_size       = 0;
  bool      is_write       = false;
  int32_t   core_id        = -1;
  std::vector<MemoryAccess*> waiting_reqs;
};

// ============================================================
// Cache — 8-way set-associative, PLRU replacement
// ============================================================
class Cache {
public:
  Cache(uint32_t bank_id, SimulationConfig config);

  // ---- input side ----
  void push_request(MemoryAccess* req);

  // ---- miss queue (to DRAM) ----
  MemoryAccess* top_miss_request();
  void          pop_miss_request();

  // ---- writeback queue (to DRAM) ----
  bool          has_wb_request();
  MemoryAccess* top_wb_request();
  void          pop_wb_request();

  // ---- DRAM response ----
  void push_memory_response(MemoryAccess* resp);

  // ---- output side (to cores) ----
  MemoryAccess* top_response();
  void          pop_response();

  // ---- clock ----
  void cycle();

  // ---- diagnostics ----
  void print_stats() const;

private:
  // Address decomposition
  static uint32_t set_index(addr_type addr) {
    return (addr >> BLOCK_BITS) & SET_MASK;
  }
  static uint64_t tag_bits(addr_type addr) {
    return addr >> (BLOCK_BITS + SET_BITS);
  }
  static addr_type block_base(addr_type addr) {
    return addr & ~(addr_type)(BLOCK_SIZE - 1);
  }

  // Core helpers
  bool     check_hit(addr_type addr, uint32_t& set_out, uint32_t& way_out);
  void     fill(addr_type baddr, bool is_write, int32_t core_id);
  void     writeback(uint32_t set_idx, uint32_t way);
  void     issue_blocks(uint32_t obj_id, PendingObject& po);

private:
  uint32_t   _bank_id;
  uint64_t   _size;        // bytes (informational)
  uint32_t   _data_width;
  cycle_type _local_cycle = 0;

  // The array — NUM_SETS × NUM_WAYS × 12 B = 2 048 × 8 × 12 = 192 KB
  std::array<CacheSet, NUM_SETS> _sets;

  // Pending fetches keyed by object id
  std::unordered_map<uint32_t, PendingObject> _pending_objects;

  std::deque<MemoryAccess*> _in_q;
  std::deque<MemoryAccess*> _miss_q;
  std::deque<MemoryAccess*> _wb_q;
  std::deque<MemoryAccess*> _out_q;

  // Stats
  uint64_t _hits       = 0;
  uint64_t _misses     = 0;
  uint64_t _read_hits  = 0;
  uint64_t _write_hits = 0;
  uint64_t _evictions  = 0;
  uint64_t _writebacks = 0;
  uint64_t _bypasses   = 0;
  uint64_t _conflicts  = 0;   // ways fully occupied in a set on fill
};