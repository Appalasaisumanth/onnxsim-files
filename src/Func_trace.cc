#include <spdlog/spdlog.h>
#include <dlfcn.h>
#include <cstring>

extern "C" {

static thread_local bool in_trace = false;

void __cyg_profile_func_enter(void* func, void* caller) {
  if (in_trace) return;
  in_trace = true;

  Dl_info info;
  if (dladdr(func, &info) && info.dli_sname) {
    if (
        strstr(info.dli_sname, "Core") ||
        strstr(info.dli_sname, "Scheduler") ||
        strstr(info.dli_sname, "Memory") ||
        strstr(info.dli_sname, "Systolic") ||
        strstr(info.dli_sname, "Operator") ||
        strstr(info.dli_sname, "Model")
       ) {
      spdlog::trace("ENTER {}", info.dli_sname);
    }
  }

  in_trace = false;
}

void __cyg_profile_func_exit(void* func, void* caller) {
  if (in_trace) return;
  in_trace = true;

  Dl_info info;
  if (dladdr(func, &info) && info.dli_sname) {
    if (
        strstr(info.dli_sname, "Core") ||
        strstr(info.dli_sname, "Scheduler") ||
        strstr(info.dli_sname, "Memory") ||
        strstr(info.dli_sname, "Systolic") ||
        strstr(info.dli_sname, "Operator") ||
        strstr(info.dli_sname, "Model")
       ) {
      spdlog::trace("EXIT  {}", info.dli_sname);
    }
  }

  in_trace = false;
}

}
