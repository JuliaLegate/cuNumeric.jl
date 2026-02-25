
## Setting Hardware Configuration

There is no programatic way to set the hardware configuration used by CuPyNumeric (as of 24.11). By default, the hardware configuration is set automatically by Legate. This configuration can be manipulated through the following environment variables:

- `LEGATE_SHOW_CONFIG` : When set to 1, the Legate config is printed to stdout
- `LEGATE_AUTO_CONFIG`: When set to 1, Legate will automatically choose the hardware configuration
- `LEGATE_CONFIG`: A string representing the hardware configuration to set

These variables must be set before launching the Julia instance running cuNumeric.jl. We recommend setting `export LEGATE_SHOW_CONFIG=1` so that the hardware configuration will be printed when Legate starts. This output is automatically captured and relayed to the user.

To manually set the hardware configuration, `export LEGATE_AUTO_CONFIG=0`, and then define your own config with something like `export LEGATE_CONFIG="--gpus 1 --cpus 10 --ompthreads 10"`. We recommend using the default memory configuration for your machine and only settings the `gpus`, `cpus` and `ompthreads`. More details about the Legate configuration can be found in the [NVIDIA Legate documentation](https://docs.nvidia.com/legate/latest/usage.html#resource-allocation). If you know where Legate is installed on your computer you can also run `legate --help` for more detailed information.