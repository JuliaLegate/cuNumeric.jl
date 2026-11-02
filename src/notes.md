We set environment variables in port.jl:
```julia
ENV["WORKER_SELF_INFO"] = "$self_addr"
ENV["WORKER_PEERS_INFO"] = "$peer_addrs"
ENV["REALM_UCP_BOOTSTRAP_PLUGIN"] = "realm_bootstrap_p2p.so"
ENV["REALM_UCP_BOOTSTRAP_MODE"] = "p2p"
```

I know these environment variables are actually being set as changing them can result in different errors.

---

**Realm code flow:**

1. https://github.com/StanfordLegion/realm/blob/main/src/realm/ucx/ucp_internal.cc#L826
```c++
boot_config.plugin_name = getenv("REALM_UCP_BOOTSTRAP_PLUGIN");
if(bootstrap_init(&boot_config, &boot_handle) != 0) {
log_ucp.error() << "failed to bootstrap ucp";
return false;
}
```

2. https://github.com/StanfordLegion/realm/blob/main/src/realm/ucx/bootstrap/bootstrap.cc#L108
```c++
case BOOTSTRAP_P2P:
if(config->plugin_name != NULL) {
    status = bootstrap_loader_init(config->plugin_name, NULL, handle);
} else {
    status = bootstrap_loader_init(BOOTSTRAP_P2P_PLUGIN, NULL, handle);
}
if(status != 0) {
    log_ucp.error() << "bootstrap_loader_init failed";
}
break;
```

3. This will dlopen the plugin file.
https://github.com/StanfordLegion/realm/blob/main/src/realm/ucx/bootstrap/bootstrap_loader.cc#L56
```c++
    int bootstrap_loader_init(const char *plugin, void *arg, bootstrap_handle_t *handle)
```

4. The plugin calls this init function:
https://github.com/StanfordLegion/realm/blob/main/src/realm/ucx/bootstrap/bootstrap_p2p.cc#L115
```c++
extern "C" int realm_ucp_bootstrap_plugin_init(void *arg, bootstrap_handle_t *handle)
{
  std::string self;
  std::vector<std::string> peers;
  if(!read_env(self, peers)) {
    std::cerr << "Failed to gather workers information " << std::endl;
    return -1;
  }

  p2p_comm = std::make_shared<p2p::P2PComm>(self, peers);
  if(!p2p_comm->Init()) { // fails to find this symbol
    std::cerr << "Failed to initialize p2p comm" << std::endl;
    return -1;
  }
...
}
```

---

**Error:**

```
From worker 5:    /tmp/conda-croot/legate/work/arch-conda/skbuild_core/_deps/realm-src/src/realm/ucx/bootstrap/bootstrap_loader.cc:64: NULL value Bootstrap unable to load 'realm_bootstrap_p2p.so'
From worker 5:            /home/david/anaconda3/envs/myenv/lib/./realm_bootstrap_p2p.so: undefined symbol: _ZN3p2p7P2PComm4InitEv
From worker 5:    Unable to create specified registered network module 'ucx'
```

The undefined symbol is:
```
c++filt -t _ZN3p2p7P2PComm4InitEv
p2p::P2PComm::Init()
```

The `realm_bootstrap_p2p.so` plugin library has **undefined symbols** for the `P2PComm` class:

```bash
$ nm -D $CONDA_PREFIX/lib/realm_bootstrap_p2p.so | grep -i p2p
                 U _ZN3p2p7P2PComm4InitEv
                 U _ZN3p2p7P2PComm8ShutdownEv
                 U _ZN3p2p7P2PComm9AllgatherEPvihS1_ih
                 U _ZN3p2p7P2PCommC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKSt6vectorIS6_SaIS6_EES8_
```

The implementation is in separate source files:
- [`p2p_comm.h`](https://github.com/StanfordLegion/realm/blob/main/src/realm/ucx/bootstrap/p2p_comm.h) / [`p2p_comm.cc`](https://github.com/StanfordLegion/realm/blob/main/src/realm/ucx/bootstrap/p2p_comm.cc) - The P2PComm class

**The problem:** The conda-built `realm_bootstrap_p2p.so` plugin only contains the `bootstrap_p2p.cc` code, but is **missing** the `p2p_comm.cc` implementation and mesh networking code.

This appears to be a **build configuration issue** in the Legate conda package (my guess).
