#include "Cache.h"
#include "spdlog/spdlog.h"
#include <cassert>

// ============================================================
// Constructor
// ============================================================
Cache::Cache(uint32_t bank_id, SimulationConfig config)
    : _bank_id(bank_id)
    , _pool(BLOCK_POOL_SIZE)
    ,_wb_pool(WB_POOL_SIZE)          //
{
    _size       = static_cast<uint64_t>(config.l2_size) * 1024ULL;
    _data_width = config.dram_req_size;

    if (_size != CACHE_SIZE_BYTES) {
        spdlog::warn("[L2 {}] config.l2_size={}KB but compiled for {}KB — "
                     "using compiled constant",
                     _bank_id, _size / 1024, CACHE_SIZE_BYTES / 1024);
        _size = CACHE_SIZE_BYTES;
    }

    for (auto& s : _sets) {
        s.plru = 0; s.valid_mask = 0;
        for (auto& e : s.ways) e.reset();
    }
    for (auto& p : _pending) {
        p.active = false;
        p.waiting_reqs.reserve(4);
    }
    _active_slots.reserve(MAX_PENDING_OBJS);
    _free_slots.resize(MAX_PENDING_OBJS);
    for (uint32_t i = 0; i < MAX_PENDING_OBJS; ++i)
        _free_slots[i] = MAX_PENDING_OBJS - 1 - i; // push in reverse so slot 0 pops first
    _id_to_slot.reserve(MAX_PENDING_OBJS * 2);

    spdlog::info("[L2 {}] {}-way, {} sets, {}B block, {}MB total | "
                 "miss_q={} pending={} pool={}",
                 _bank_id, NUM_WAYS, NUM_SETS, BLOCK_SIZE,
                 _size / (1024 * 1024),
                 MAX_MISS_Q_DEPTH, MAX_PENDING_OBJS, BLOCK_POOL_SIZE);
}

// ============================================================
// Queue API
// ============================================================
void Cache::push_request(MemoryAccess* req) { _in_q.push_back(req); }

MemoryAccess* Cache::top_miss_request() { return _miss_q.empty() ? nullptr : _miss_q.front(); }
void          Cache::pop_miss_request() { if (!_miss_q.empty()) _miss_q.pop_front(); }

bool          Cache::has_wb_request()   { return !_wb_q.empty(); }
MemoryAccess* Cache::top_wb_request()   { return _wb_q.empty() ? nullptr : _wb_q.front(); }
void          Cache::pop_wb_request()   { if (!_wb_q.empty()) _wb_q.pop_front(); }

MemoryAccess* Cache::top_response()     { return _out_q.empty() ? nullptr : _out_q.front(); }
void          Cache::pop_response()     { if (!_out_q.empty()) _out_q.pop_front(); }

// ============================================================
// Pending slot management
// ============================================================
uint32_t Cache::find_pending_slot(uint32_t obj_id) const
{
    auto it = _id_to_slot.find(obj_id);
    return (it != _id_to_slot.end()) ? it->second : MAX_PENDING_OBJS;
}

uint32_t Cache::alloc_pending_slot()
{
    if (_free_slots.empty()) return MAX_PENDING_OBJS;
    uint32_t slot = _free_slots.back();
    _free_slots.pop_back();
    return slot;
}

void Cache::free_pending_slot(uint32_t slot)
{
    _id_to_slot.erase(_pending[slot].original_id);
    _pending[slot].reset();
    _free_slots.push_back(slot);

    // O(1) removal from _active_slots using stored index
    uint32_t idx = _pending[slot].active_slots_idx;
    if (idx < _active_slots.size()) {
        _active_slots[idx] = _active_slots.back();
        // Update the moved slot's stored index
        _pending[_active_slots[idx]].active_slots_idx = idx;
        _active_slots.pop_back();
    }
    _active_pending--;
}
uint32_t Cache::compute_burst_cycles(uint32_t obj_size_bytes)
{
    uint32_t transfer = (obj_size_bytes + HBM_BYTES_PER_CYCLE - 1) / HBM_BYTES_PER_CYCLE;
    return HBM_MIN_LATENCY_CYCLES + transfer;
}

// ============================================================
// check_hit
// ============================================================
bool Cache::check_hit(addr_type addr, uint32_t& si, uint32_t& way)
{
    si         = set_index(addr);
    uint64_t t = tag_bits(addr);
    way        = _sets[si].find_tag(t);
    return way < NUM_WAYS;
}

// ============================================================
// fill
// ============================================================
void Cache::fill(addr_type baddr, bool is_write, int32_t core_id)
{
    uint32_t  si  = set_index(baddr);
    uint64_t  tag = tag_bits(baddr);
    CacheSet& s   = _sets[si];

    uint32_t w = s.find_tag(tag);
    if (w < NUM_WAYS) {
        if (is_write) s.ways[w].dirty = 1;
        s.touch(w);
        return;
    }

    w = s.free_way();
    if (w == NUM_WAYS) {
        _conflicts++;
        w = s.lru_way();
        writeback(si, w);
        _evictions++;
        s.clear_valid(w);
    }

    CacheEntry& e = s.ways[w];
    e.tag   = tag;
    e.valid = 1;
    e.dirty = is_write ? 1 : 0;
    e.set_core(core_id);
    s.set_valid(w);
    s.touch(w);
}

// ============================================================
// writeback
// ============================================================
void Cache::writeback(uint32_t si, uint32_t w)
{
    CacheEntry& e = _sets[si].ways[w];
    if (!e.valid) { e.reset(); return; }

    if (e.dirty) {
        addr_type baddr = (static_cast<addr_type>(e.tag) << (BLOCK_BITS + SET_BITS))
                        | (static_cast<addr_type>(si)    <<  BLOCK_BITS);

        MemoryAccess* wb = _wb_pool.alloc(); 
        wb->dram_address = baddr;
        wb->size         = BLOCK_SIZE;
        wb->object_size  = BLOCK_SIZE;
        wb->write        = true;
        wb->request      = true;
        wb->core_id      = e.get_core();
        wb->buffer_id    = 0;
        wb->id           = 0;
        _wb_q.push_back(wb);
        _writebacks++;
    }
    e.reset();
}

// ============================================================
// issue_blocks
//
// Issues the next batch of 128B block requests for a pending
// object into _miss_q.
//
// CRITICAL FOR PERFORMANCE:
// We issue ALL remaining blocks in one call (up to miss_q space).
// This means a 786KB object (6144 blocks) is fully queued to HBM
// in a single issue_blocks() call instead of dripping one per cycle.
// The HBM controller then models the bandwidth correctly via its
// own timing model — we don't need to serialize it here.
//
// Each block gets slot index as its id for response routing.
// ============================================================
void Cache::issue_blocks(uint32_t slot, PendingObject& po)
{
    uint32_t available = MAX_MISS_Q_DEPTH - static_cast<uint32_t>(_miss_q.size());
    uint32_t remaining = po.total_blocks - po.issued_blocks;
    uint32_t to_issue  = std::min(available, remaining);

    while (to_issue-- > 0) {
        addr_type sub_addr = po.base_addr
                           + static_cast<addr_type>(po.issued_blocks) * BLOCK_SIZE;
        MemoryAccess* sub = _pool.alloc();
        // pool fallback detection is the same
        sub->id           = slot;
        sub->dram_address = sub_addr;
        sub->size         = BLOCK_SIZE;
        sub->object_size  = BLOCK_SIZE;
        sub->write        = po.is_write;
        sub->request      = true;
        sub->core_id      = po.core_id;
        sub->buffer_id    = 0;
        _miss_q.push_back(sub);
        po.issued_blocks++;
    }
}

// ============================================================
// push_memory_response
//
// Called when HBM returns one 128B block.
// resp->id == slot index in _pending[].
//
// For each returned block:
//   1. Fill the cache line
//   2. Increment returned_blocks counter
//   3. If all blocks returned, wake waiting cores
//
// This correctly models real hardware: cores are released only
// after all cache lines for their request have been filled.
// ============================================================
void Cache::push_memory_response(MemoryAccess* resp)
{
    uint32_t slot = resp->id;

    if (slot >= MAX_PENDING_OBJS || !_pending[slot].active) {
        // Response for a slot that was freed or is invalid.
        // This can happen for writeback completions (id=0).
        _pool.release(resp);
        return;
    }

    PendingObject& po = _pending[slot];

    // Fill this cache line
    fill(block_base(resp->dram_address), po.is_write, po.core_id);
    po.returned_blocks++;

    // Continue issuing remaining blocks if any and queue has room
    if (po.issued_blocks < po.total_blocks &&
        _miss_q.size() < MAX_MISS_Q_DEPTH) {
        issue_blocks(slot, po);
    }

    // All blocks returned — release waiting cores
    if (po.returned_blocks >= po.total_blocks) {
        for (MemoryAccess* req : po.waiting_reqs) {
            req->request = false;
            _out_q.push_back(req);
        }
        free_pending_slot(slot);
    }

    _pool.release(resp);
}

// ============================================================
// process_one_request
//
// Takes one request from _in_q and processes it:
//   HIT  → immediate response
//   MISS → create/join PendingObject, issue blocks to HBM
// ============================================================
bool Cache::process_one_request()
{
    if (_in_q.empty()) return false;

    MemoryAccess* req = _in_q.front();
    _in_q.pop_front();

    addr_type addr = req->dram_address;
    uint32_t  si, w;

    // ---- HIT ----
    if (check_hit(addr, si, w)) {
        _hits++;
        if (req->write) {
            _write_hits++;
            _sets[si].ways[w].dirty = 1;
        } else {
            _read_hits++;
        }
        _sets[si].touch(w);
        req->request = false;
        _out_q.push_back(req);
        return true;
    }

    // ---- MISS ----
    _misses++;

    uint32_t  obj_id   = req->id;
    uint32_t  n_blocks = (req->object_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    addr_type base     = block_base(addr);

    // Coalesce: same object already in-flight — just add to waiters
    uint32_t slot = find_pending_slot(obj_id);
    if (slot < MAX_PENDING_OBJS) {
        _pending[slot].waiting_reqs.push_back(req);
        return true;
    }

    // New miss — allocate a pending slot
    slot = alloc_pending_slot();
    if (slot == MAX_PENDING_OBJS) {
        // All slots full — re-queue and stall, but keep processing others
        _in_q.push_front(req);
        return true; // return true so cycle() keeps draining other reqs
    }
    _pending[slot].active_slots_idx = static_cast<uint32_t>(_active_slots.size());
_active_slots.push_back(slot);
_id_to_slot[obj_id] = slot;  

    PendingObject& po  = _pending[slot];
    po.total_blocks    = n_blocks;
    po.issued_blocks   = 0;
    po.returned_blocks = 0;
    po.base_addr       = base;
    po.obj_size        = req->object_size;
    po.is_write        = req->write;
    po.core_id         = req->core_id;
    po.original_id     = obj_id;
    po.active          = true;
    po.waiting_reqs.clear();
    po.waiting_reqs.push_back(req);
    _active_pending++;

    if (req->object_size >= LARGE_OBJ_THRESHOLD) {
        po.is_burst_mode   = true;
        po.burst_countdown = compute_burst_cycles(req->object_size);
        po.issued_blocks   = po.total_blocks; // mark as "all issued" — no block tracking

        // Issue ONE MemoryAccess representing the entire object burst.
        // The HBM controller sees object_size and models BW correctly.
        MemoryAccess* burst = _pool.alloc();
        burst->id           = slot;
        burst->dram_address = base;
        burst->size         = req->object_size; // full object, not BLOCK_SIZE
        burst->object_size  = req->object_size;
        burst->write        = po.is_write;
        burst->request      = true;
        burst->core_id      = po.core_id;
        burst->buffer_id    = 0;
        _miss_q.push_back(burst);
        // NOTE: We do NOT fill cache lines here — we fill them in cycle()
        // once burst_countdown reaches zero, modeling the pipeline delay.

    } else {
        // ---- SMALL OBJECT: existing per-block mode ----
        po.is_burst_mode = false;
        issue_blocks(slot, po);
    }

    return true;

    // Issue ALL blocks immediately — fills miss_q up to MAX_MISS_Q_DEPTH.
    // For a 786KB object (6144 blocks) with MAX_MISS_Q_DEPTH=65536,
    // all 6144 blocks are queued in one call. The HBM controller
    // will pipeline them at full bandwidth. No drip-feeding needed.
    issue_blocks(slot, po);

    return true;
}

// ============================================================
// cycle
// ============================================================
void Cache::cycle()
{
    _local_cycle++;

    // Continue issuing blocks for any pending objects that still
    // have unissued blocks (only happens if miss_q was full last time)
    if (_active_pending > 0 && _miss_q.size() < MAX_MISS_Q_DEPTH) {
        for (uint32_t slot : _active_slots) {
            if (_miss_q.size() >= MAX_MISS_Q_DEPTH) break;
            PendingObject& po = _pending[slot];
            if (po.active && po.issued_blocks < po.total_blocks)
                issue_blocks(slot, po);
        }
    }

    // Process new input requests — up to MAX_IN_PER_CYCLE per cycle
    // Only accept new requests if pending pool has headroom
    if (_active_pending < MAX_PENDING_OBJS - 16) {
        uint32_t processed = 0;
        while (processed < MAX_IN_PER_CYCLE && !_in_q.empty()) {
            if (!process_one_request()) break;
            ++processed;
        }
    } else {
        // Pool near full — only process requests that will coalesce
        // with existing pending objects (hits or coalescences)
        uint32_t processed = 0;
        while (processed < MAX_IN_PER_CYCLE && !_in_q.empty()) {
            MemoryAccess* req = _in_q.front();
            uint32_t si, w;
            if (check_hit(req->dram_address, si, w) ||
                find_pending_slot(req->id) < MAX_PENDING_OBJS) {
                process_one_request();
                processed++;
            } else {
                break; // new miss would need a slot — wait
            }
        }
    }
}

// ============================================================
// print_stats
// ============================================================
void Cache::print_stats() const
{
    uint64_t total    = _hits + _misses;
    double   hit_rate = total ? static_cast<double>(_hits) / total : 0.0;

    uint64_t occupied = 0;
    for (const auto& s : _sets)
        occupied += __builtin_popcount(s.valid_mask);

    double util = 100.0 * static_cast<double>(occupied)
                        / static_cast<double>(NUM_SETS * NUM_WAYS);

    spdlog::info("===== L2 BANK {} ({}-way, {} sets) =====",
                 _bank_id, NUM_WAYS, NUM_SETS);
    spdlog::info("  Cycles        : {}", _local_cycle);
    spdlog::info("  Hits          : {}", _hits);
    spdlog::info("  Misses        : {}", _misses);
    spdlog::info("  Hit rate      : {:.4f}", hit_rate);
    spdlog::info("  Read hits     : {}", _read_hits);
    spdlog::info("  Write hits    : {}", _write_hits);
    spdlog::info("  Evictions     : {}", _evictions);
    spdlog::info("  Conflicts     : {}", _conflicts);
    spdlog::info("  Writebacks    : {}", _writebacks);
    spdlog::info("  Pool fallback : {}", _pool_fallback);
    spdlog::info("  Cache util    : {:.1f}% ({}/{} blocks)",
                 util, occupied, NUM_SETS * NUM_WAYS);
    spdlog::info("  Active pending: {}", _active_pending);
    spdlog::info("  Miss Q depth  : {}", MAX_MISS_Q_DEPTH);
    spdlog::info("  Pool size     : {}", BLOCK_POOL_SIZE);
}