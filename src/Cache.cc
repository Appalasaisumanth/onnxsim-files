#include "Cache.h"
#include "spdlog/spdlog.h"

// ============================================================
// Constructor
// ============================================================
Cache::Cache(uint32_t bank_id, SimulationConfig config)
    : _bank_id(bank_id)
{
  _size       = (uint64_t)config.l2_size * 1024ULL;
  _data_width = config.dram_req_size;

  if (_size != CACHE_SIZE_BYTES) {
    spdlog::warn("[L2 {}] config.l2_size={}KB but compiled for {}KB — "
                 "using compiled constant",
                 _bank_id, _size / 1024, CACHE_SIZE_BYTES / 1024);
    _size = CACHE_SIZE_BYTES;
  }

  for (auto& s : _sets)
    for (auto& w : s.ways) w.reset();

  spdlog::info("[L2 {}] {}-way set-associative, {} sets, block={}B, "
               "total={}KB, entry={}B, array={}KB, PLRU={}B",
               _bank_id, NUM_WAYS, NUM_SETS, BLOCK_SIZE,
               _size / 1024,
               sizeof(CacheEntry),
               (NUM_SETS * NUM_WAYS * sizeof(CacheEntry)) / 1024,
               NUM_SETS * sizeof(uint8_t));
}

// ============================================================
// Queue API
// ============================================================
void          Cache::push_request(MemoryAccess* req) { _in_q.push_back(req); }

MemoryAccess* Cache::top_miss_request()  { return _miss_q.empty() ? nullptr : _miss_q.front(); }
void          Cache::pop_miss_request()  { if (!_miss_q.empty()) _miss_q.pop_front(); }

bool          Cache::has_wb_request()    { return !_wb_q.empty(); }
MemoryAccess* Cache::top_wb_request()    { return _wb_q.empty()  ? nullptr : _wb_q.front(); }
void          Cache::pop_wb_request()    { if (!_wb_q.empty()) _wb_q.pop_front(); }

MemoryAccess* Cache::top_response()      { return _out_q.empty() ? nullptr : _out_q.front(); }
void          Cache::pop_response()      { if (!_out_q.empty()) _out_q.pop_front(); }

// ============================================================
// check_hit
// ============================================================
bool Cache::check_hit(addr_type addr, uint32_t& set_out, uint32_t& way_out)
{
  set_out = set_index(addr);
  uint64_t tag = tag_bits(addr);
  way_out = _sets[set_out].find_tag(tag);
  return way_out < NUM_WAYS;
}

// ============================================================
// fill
// ============================================================
void Cache::fill(addr_type baddr, bool is_write, int32_t core_id)
{
  uint32_t si  = set_index(baddr);
  uint64_t tag = tag_bits(baddr);
  CacheSet& s  = _sets[si];

  // Already present — just mark dirty and refresh PLRU
  uint32_t w = s.find_tag(tag);
  if (w < NUM_WAYS) {
    if (is_write) s.ways[w].dirty = 1;
    s.touch(w);
    return;
  }

  // Pick a free way, else evict PLRU victim
  w = s.free_way();
  if (w == NUM_WAYS) {
    _conflicts++;
    w = s.lru_way();
    writeback(si, w);
    _evictions++;
  }

  CacheEntry& e = s.ways[w];
  e.tag   = tag;
  e.valid = 1;
  e.dirty = is_write ? 1 : 0;
  e.set_core(core_id);

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
    addr_type baddr = ((addr_type)e.tag << (BLOCK_BITS + SET_BITS))
                    | ((addr_type)si     <<  BLOCK_BITS);

    MemoryAccess* wb = new MemoryAccess();
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
// ============================================================
void Cache::issue_blocks(uint32_t obj_id, PendingObject& po)
{
  while (po.issued_blocks < po.total_blocks &&
         _miss_q.size() < MAX_MISS_Q_DEPTH)
  {
    uint32_t  i        = po.issued_blocks;
    addr_type sub_addr = po.base_addr + (addr_type)i * BLOCK_SIZE;

    MemoryAccess* sub = new MemoryAccess();
    sub->id           = obj_id;
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
// ============================================================
void Cache::push_memory_response(MemoryAccess* resp)
{
  uint32_t obj_id = resp->id;

  auto it = _pending_objects.find(obj_id);
  if (it == _pending_objects.end()) {
    delete resp;
    return;
  }

  PendingObject& po = it->second;

  fill(block_base(resp->dram_address), resp->write, resp->core_id);

  po.returned_blocks++;

  if (po.issued_blocks < po.total_blocks)
    issue_blocks(obj_id, po);

  if (po.returned_blocks >= po.total_blocks) {
    for (auto* req : po.waiting_reqs) {
      req->request = false;
      _out_q.push_back(req);
    }
    _pending_objects.erase(it);
  }

  delete resp;
}

// ============================================================
// cycle
// ============================================================
void Cache::cycle()
{
  _local_cycle++;

  // Keep drip-feeding stalled pending objects
  for (auto& [id, po] : _pending_objects) {
    if (po.issued_blocks < po.total_blocks &&
        _miss_q.size() < MAX_MISS_Q_DEPTH)
    {
      issue_blocks(id, po);
    }
  }

  if (_in_q.empty()) return;

  MemoryAccess* req = _in_q.front();
  _in_q.pop_front();

  addr_type addr = req->dram_address;
  uint32_t  si, w;

  // ---- HIT ----
  if (check_hit(addr, si, w)) {
    _hits++;
    if (req->write) { _write_hits++; _sets[si].ways[w].dirty = 1; }
    else              _read_hits++;

    _sets[si].touch(w);      // 3 bit-flips, O(1)

    req->request = false;
    _out_q.push_back(req);
    return;
  }

  // ---- MISS ----
  _misses++;

  uint32_t  obj_id   = req->id;
  uint32_t  n_blocks = (req->object_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  addr_type base     = block_base(addr);

  if ((uint64_t)n_blocks * BLOCK_SIZE > _size) {
    _bypasses++;
    spdlog::warn("[L2 {}] Bypass: obj_id={} size={}B > cache {}KB",
                 _bank_id, obj_id, req->object_size, _size / 1024);
  }

  // Coalesce into an existing in-flight fetch for the same object
  auto it = _pending_objects.find(obj_id);
  if (it != _pending_objects.end()) {
    it->second.waiting_reqs.push_back(req);
    return;
  }

  // First miss — create pending object and start issuing
  PendingObject po;
  po.total_blocks    = n_blocks;
  po.issued_blocks   = 0;
  po.returned_blocks = 0;
  po.base_addr       = base;
  po.obj_size        = req->object_size;
  po.is_write        = req->write;
  po.core_id         = req->core_id;
  po.waiting_reqs.push_back(req);

  _pending_objects[obj_id] = std::move(po);
  issue_blocks(obj_id, _pending_objects[obj_id]);
}

// ============================================================
// print_stats
// ============================================================
void Cache::print_stats() const
{
  uint64_t total    = _hits + _misses;
  double   hit_rate = total ? (double)_hits / total : 0.0;

  uint64_t occupied = 0;
  for (const auto& s : _sets)
    for (const auto& ew : s.ways)
      if (ew.valid) occupied++;

  double util = 100.0 * occupied / (NUM_SETS * NUM_WAYS);

  spdlog::info("===== L2 BANK {} ({}-way, {} sets) =====", _bank_id, NUM_WAYS, NUM_SETS);
  spdlog::info("  Cycles        : {}", _local_cycle);
  spdlog::info("  Hits          : {}", _hits);
  spdlog::info("  Misses        : {}", _misses);
  spdlog::info("  Hit rate      : {:.4f}", hit_rate);
  spdlog::info("  Read hits     : {}", _read_hits);
  spdlog::info("  Write hits    : {}", _write_hits);
  spdlog::info("  Evictions     : {}", _evictions);
  spdlog::info("  Conflicts     : {}", _conflicts);
  spdlog::info("  Writebacks    : {}", _writebacks);
  spdlog::info("  Bypasses      : {}", _bypasses);
  spdlog::info("  Cache util    : {:.1f}% ({}/{} blocks)",
               util, occupied, NUM_SETS * NUM_WAYS);
  spdlog::info("  Pending objs  : {}", _pending_objects.size());
}