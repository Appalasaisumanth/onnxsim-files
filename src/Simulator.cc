#include "Simulator.h"

#include <filesystem>
#include <string>

#include "SystolicOS.h"
#include "SystolicWS.h"


namespace fs = std::filesystem;

Simulator::Simulator(SimulationConfig config, bool language_mode)
    : _config(config), _core_cycles(0), _language_mode(language_mode) {
  // Create dram object
  spdlog::info("Simulator Configuration:");
  for (int i=0; i<config.num_cores;i++)
    spdlog::info("[Core {}] Systolic Array Throughput: {} GFLOPS, Spad size: {} KB, Accumulator size: {} KB",
      i, config.max_systolic_flops(i), config.core_config[i].spad_size, config.core_config[i].accum_spad_size);
  spdlog::info("DRAM Bandwidth {} GB/s", config.max_dram_bandwidth());
  _core_period = 1000000 / (config.core_freq);
  _icnt_period = 1000000 / (config.icnt_freq);
  _dram_period = 1000000 / (config.dram_freq);
  _core_time = 0;
  _dram_time = 0;
  _icnt_time = 0;

  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  if (config.dram_type == DramType::SIMPLE) {
    _dram = std::make_unique<SimpleDram>(config);
  } else if (config.dram_type == DramType::RAMULATOR1) {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator>(config);
  } 
  else if (config.dram_type == DramType::RAMULATOR2) 
  {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator2 config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator2>(config);
  } 
  else {
    spdlog::error("[Configuration] Invalid DRAM type...!");
    exit(EXIT_FAILURE);
  }


  // Create interconnect object
  if (config.icnt_type == IcntType::SIMPLE) {
    _icnt = std::make_unique<SimpleInterconnect>(config);
  } else if (config.icnt_type == IcntType::BOOKSIM2) {
    _icnt = std::make_unique<Booksim2Interconnect>(config);
  } else {
    spdlog::error("[Configuration] {} Invalid interconnect type...!");
    exit(EXIT_FAILURE);
  }
  _icnt_interval = config.icnt_print_interval;
  spdlog::info("ICNT print interval: {} cycles", _icnt_interval);
 spdlog::info("num cores : [{}]",config.num_cores);
  // Create core objects
  _cores.resize(config.num_cores);
  _n_cores = config.num_cores;
  _n_memories = config.dram_channels;
  _noc_node_per_core = config.icnt_injection_ports_per_core;
  _memory_req_size = config.dram_req_size;
    _ins_num.resize(config.num_cores);
  _tile_num.resize(config.num_cores);
  for (int core_index = 0; core_index < _n_cores; core_index++) {
    _cores[core_index] = Core::create(core_index, config);
    _ins_num[core_index]=0;
    _tile_num[core_index]=0;

  }

   _use_l2 = _config.use_l2;

if (_use_l2) {

  _l2_banks.resize(_config.dram_channels);

  for (uint32_t i = 0;
       i < _config.dram_channels;
       i++) {

    _l2_banks[i] =
      std::make_unique<Cache>(
          i, _config);
  }
}



  //Configure Hardware Scheduler
  _scheduler = Scheduler::create(_config, &_core_cycles, &_core_time, this);
  
  /* Create heap */
  std::make_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::run_simulator() {
  spdlog::info("has_l2,[{}]",_use_l2);
  spdlog::info("======Start Simulation=====");
  cycle();
}

void Simulator::handle_model() {
  if(_language_mode) {
    _lang_scheduler->cycle();
    if(_lang_scheduler->can_schedule_model()) {
      _models.push_back(_lang_scheduler->pop_model());
      std::push_heap(_models.begin(), _models.end(), CompareModel());
    }
  }
  while (!_models.empty() && _models.front()->get_request_time() <= _core_time) {
    std::unique_ptr<Model> launch_model = std::move(_models.front());
    std::pop_heap(_models.begin(), _models.end(), CompareModel());
    _models.pop_back();

    launch_model->initialize_model(_weight_table[launch_model->get_name()]);
    launch_model->set_request_time(_core_time);
    spdlog::info("Schedule model: {} at {} us", launch_model->get_name(), _core_time);
    _scheduler->schedule_model(std::move(launch_model), 1);
  }
}
void Simulator::cycle() {
  OpStat op_stat; 
  ModelStat model_stat; 
  uint32_t tile_count; 
  bool is_accum_tile;
  while (running()) {
    int model_id = 0;
    set_cycle_mask();

    if (_cycle_mask & CORE_MASK) {
      handle_model();
      for (int core_id = 0; core_id < _n_cores; core_id++) {
        std::unique_ptr<Tile> finished_tile = _cores[core_id]->pop_finished_tile();
        if (finished_tile->status == Tile::Status::FINISH) _scheduler->finish_tile(core_id, finished_tile->layer_id);
        if (!_scheduler->empty()) {
          is_accum_tile = _scheduler->is_accum_tile(core_id, 0);
          if (_cores[core_id]->can_issue(is_accum_tile)) {
            std::unique_ptr<Tile> tile = _scheduler->get_tile(core_id);
            if (tile->status == Tile::Status::INITIALIZED) {
              _cores[core_id]->issue(std::move(tile));
              _tile_timestamp.push_back(std::chrono::high_resolution_clock::now());
            }
          }
        }
        _cores[core_id]->cycle();
      }
      _core_cycles++;
    }

    if (_cycle_mask & DRAM_MASK) {
      _dram->cycle();
      if (_use_l2) for (auto& l2 : _l2_banks) l2->cycle();
    }

      if (_cycle_mask & ICNT_MASK) {
        _icnt_cycle++;

        for (int core_id = 0; core_id < _n_cores; core_id++) {
          for (int noc_id = 0; noc_id < _noc_node_per_core; noc_id++) {
            int port_id = core_id * _noc_node_per_core + noc_id;
            if (_cores[core_id]->has_memory_request()) {
              MemoryAccess* front = _cores[core_id]->top_memory_request();
              object_size_hist[front->object_size]++;
              front->core_id = core_id;
              if (!_icnt->is_full(port_id, front)) {
                _icnt->push(port_id, get_dest_node(front), front);
                _cores[core_id]->pop_memory_request();
                _nr_from_core++;
              }
            }
            if (!_icnt->is_empty(port_id)) {
              _cores[core_id]->push_memory_response(_icnt->top(port_id));
              _icnt->pop(port_id);
              _nr_to_core++;
            }
          }
        }

        int core_offset = _n_cores * _noc_node_per_core;

        if (_use_l2) 
        {
           for (int mem_id = 0; mem_id < _n_memories; mem_id++) 
           {

                    Cache* l2 = _l2_banks[mem_id].get();

                    // ICNT → L2
                    if (!_icnt->is_empty(core_offset + mem_id)) {
                      l2->push_request(_icnt->top(core_offset + mem_id));
                      _icnt->pop(core_offset + mem_id);
                    }

                    // L2 MISS → DRAM (1/cycle)
                    if (l2->top_miss_request()) {
                      auto* miss = l2->top_miss_request();
                      if (!_dram->is_full(mem_id, miss)) {
                        _dram->push(mem_id, miss);
                        l2->pop_miss_request();
                      }
                    }

                    // L2 WB → DRAM (1/cycle)
                    if (l2->has_wb_request()) {
                      auto* wb = l2->top_wb_request();
                      if (!_dram->is_full(mem_id, wb)) {
                        _dram->push(mem_id, wb);
                        l2->pop_wb_request();
                      }
                    }

                    // DRAM → L2 (1/cycle)
                    if (!_dram->is_empty(mem_id)) {
                      auto* resp = _dram->top(mem_id);
                      l2->push_memory_response(resp);
                      _dram->pop(mem_id);
                    }

                    // L2 → ICNT (1/cycle)
                    if (l2->top_response()) 
                    {

                        auto* resp = l2->top_response();
                        if (resp->core_id>=_n_cores || resp->core_id<0) {
                          spdlog::error("Invalid core id {} in response", resp->core_id);
                          continue;
                        }
                        if (!_icnt->is_full(core_offset + mem_id, resp)) {
                          _icnt->push(core_offset + mem_id,
                                      get_dest_node(resp),
                                      resp);
                          l2->pop_response();
                        }
                    }
                  
        
        } 
      } 
        else {
          for (int mem_id = 0; mem_id < _n_memories; mem_id++) {
            if (!_icnt->is_empty(core_offset + mem_id) && !_dram->is_full(mem_id, _icnt->top(core_offset + mem_id))) {
              _dram->push(mem_id, _icnt->top(core_offset + mem_id));
              _icnt->pop(core_offset + mem_id);
              _nr_to_mem++;
            }
            if (!_dram->is_empty(mem_id) && !_icnt->is_full(core_offset + mem_id, _dram->top(mem_id))) {
              _icnt->push(core_offset + mem_id, get_dest_node(_dram->top(mem_id)), _dram->top(mem_id));
              _dram->pop(mem_id);
              _nr_from_mem++;
            }
          }
        }

        if (_icnt_interval>0 && _icnt_cycle % _icnt_interval == 0) {
          // spdlog::info("[ICNT] Core->ICNT request {}GB/Sec,{}", ((_memory_req_size * _nr_from_core * (1000 / _icnt_period) / _icnt_interval)), _nr_from_core);
          // spdlog::info("[ICNT] Core<-ICNT request {}GB/Sec,{}", ((_memory_req_size * _nr_to_core * (1000 / _icnt_period) / _icnt_interval)), _nr_to_core);
          // spdlog::info("[ICNT] ICNT->MEM request {}GB/Sec,{}", ((_memory_req_size * _nr_to_mem * (1000 / _icnt_period) / _icnt_interval)), _nr_to_mem);
          // spdlog::info("[ICNT] ICNT<-MEM request {}GB/Sec,{}", ((_memory_req_size * _nr_from_mem * (1000 / _icnt_period) / _icnt_interval)), _nr_from_mem);
          //_nr_from_core = 0; _nr_to_core = 0; _nr_to_mem = 0; _nr_from_mem = 0;
        }

        _icnt->cycle();
      }
    }
  

  spdlog::info("Simulation Finished at {} cycle {} us", _core_cycles, _core_cycles / (_config.core_freq));
   spdlog::info("[ICNT] Core->ICNT request {}GB/Sec,{}", ((_memory_req_size * _nr_from_core * (1000 / _icnt_period) / _icnt_interval)), _nr_from_core);
          spdlog::info("[ICNT] Core<-ICNT request {}GB/Sec,{}", ((_memory_req_size * _nr_to_core * (1000 / _icnt_period) / _icnt_interval)), _nr_to_core);
          spdlog::info("[ICNT] ICNT->MEM request {}GB/Sec,{}", ((_memory_req_size * _nr_to_mem * (1000 / _icnt_period) / _icnt_interval)), _nr_to_mem);
          spdlog::info("[ICNT] ICNT<-MEM request {}GB/Sec,{}", ((_memory_req_size * _nr_from_mem * (1000 / _icnt_period) / _icnt_interval)), _nr_from_mem);


  for (int core_id = 0; core_id < _n_cores; core_id++) _cores[core_id]->print_stats();
  

  _icnt->print_stats();
  _dram->print_stat();
   if (_use_l2) for (auto& l2 : _l2_banks) l2->print_stats();

std::cout << "Object Size Histogram\n";

long long cumulative_sum = 0;

for (auto &it : object_size_hist)
{
    cumulative_sum += (long long)it.first * it.second;

    std::cout << it.first << " bytes : "
              << it.second
              << "  cumulative_bytes: "
              << cumulative_sum
              << std::endl;
}
}

void Simulator::register_model(std::unique_ptr<Model> model) {
  if(_weight_table.find(model->get_name()) == _weight_table.end()) {
    model->initialize_weight(_weight_table[model->get_name()]);
  } 
  _models.push_back(std::move(model));
  std::push_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::register_language_model(json info, std::unique_ptr<LanguageModel> model) {
  std::string name = info["name"];
  std::string trace_file = info["trace_file"];
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  trace_file = fs::path(onnxim_path).append("traces").append(trace_file).string();
  if(_weight_table.find(name) == _weight_table.end()) {
    model->initialize_weight(_weight_table[name]);
  }
  _lang_scheduler = LangScheduler::create(name, trace_file, std::move(model), _config, info);
}

void Simulator::finish_language_model(uint32_t model_id) {
  _lang_scheduler->finish_model(model_id);
}

bool Simulator::running() {
  bool running = false;
  running |= !_models.empty();
  for (auto &core : _cores) {
    running = running || core->running();
  }
  running = running || _icnt->running();
  running = running || _dram->running();
  running = running || !_scheduler->empty();
  if(_language_mode) {
    running = running || _lang_scheduler->busy();
  }
  return running;
}

void Simulator::set_cycle_mask() {
  _cycle_mask = 0x0;
  uint64_t minimum_time = MIN3(_core_time, _dram_time, _icnt_time);
  if (_core_time <= minimum_time) {
    _cycle_mask |= CORE_MASK;
    _core_time += _core_period;
  }
  if (_dram_time <= minimum_time) {
    _cycle_mask |= DRAM_MASK;
    _dram_time += _dram_period;
  }
  if (_icnt_time <= minimum_time) {
    _cycle_mask |= ICNT_MASK;
    _icnt_time += _icnt_period;
  }
}

uint32_t Simulator::get_dest_node(MemoryAccess *access) {
  if (access->request) {
    return _config.num_cores * _config.icnt_injection_ports_per_core + _dram->get_channel_id(access);
  } else {
    return access->core_id * _config.icnt_injection_ports_per_core + (_dram->get_channel_id(access) % _config.icnt_injection_ports_per_core);
  }
}

const double Simulator::get_tile_ops() {
  std::chrono::duration<double> duration = _tile_timestamp.back() - _tile_timestamp.front();
  spdlog::info("no.of tiles :[{}]",duration.count());
  if (_tile_timestamp.empty())
    return 0.0;
  else
    return _tile_timestamp.size() / duration.count();
}
