#include <riscv/sim.h>

// Copied from spike main.
// TODO: This should really be provided in libriscv
static std::vector<std::pair<reg_t, abstract_mem_t*>> make_mems(const std::vector<mem_cfg_t> &layout)
{
  std::vector<std::pair<reg_t, abstract_mem_t*>> mems;
  mems.reserve(layout.size());
  for (const auto &cfg : layout) {
    mems.push_back(std::make_pair(cfg.get_base(), new mem_t(cfg.get_size())));
  }
  return mems;
}

int main()
{
  std::vector<mem_cfg_t> mem_cfg { mem_cfg_t(0x80000000, 0x10000000) };
  std::vector<size_t> hartids = {0};
  cfg_t cfg(std::make_pair(0, 0),
            nullptr,
            "rv64gcv",
            "MSU",
            "vlen:128,elen:64",
            false,
            endianness_little,
            16,
            (1 << PMP_SHIFT),
            mem_cfg,
            hartids,
            false,
            4);
  std::vector<device_factory_t*> plugin_devices;
  std::vector<std::string> htif_args {"pk", "hello"};
  debug_module_config_t dm_config = {
    .progbufsize = 2,
    .max_sba_data_width = 0,
    .require_authentication = false,
    .abstract_rti = 0,
    .support_hasel = true,
    .support_abstract_csr_access = true,
    .support_abstract_fpr_access = true,
    .support_haltgroups = true,
    .support_impebreak = true
  };
  std::vector<std::pair<reg_t, abstract_mem_t*>> mems =
      make_mems(cfg.mem_layout());
  sim_t sim(&cfg, false,
            mems,
            plugin_devices,
            htif_args,
            dm_config,
            nullptr,
            true,
            nullptr,
            false,
            nullptr);
  sim.run();
}
