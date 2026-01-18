# GTCRN LADSPA Plugin (ONNX Runtime)

This is a LADSPA plugin for real-time speech enhancement using the GTCRN model.

## Features

- **High Performance**: Uses ONNX Runtime.
- **Portable**: Can be built as a single, static `.so` file with **no external dependencies**.
- **Real-time**: Designed for low-latency audio processing.

## Building

### Minimal Build (Static Link - Docker)

This produces the smallest, single-file plugin with no dependencies (~12MB). This is the most robust option for distribution.

1. **Build Minimal Runtime**:
   ```bash
   ./build-minimal-docker.sh
   ```
   *This takes 5-20 minutes on first run.*

2. **Build Plugin**:
   ```bash
   ./build.sh minimal
   ```

### Static Build (Download - Bundled)

This downloads the official ONNX Runtime from Microsoft and bundles it. Easier than Docker but larger (~22MB).

```bash
./build.sh static
```

### Dynamic Build (System Lib)

Fastest for development. Requires `onnxruntime` installed on the system.

```bash
./build.sh dynamic
```

## PipeWire Configuration

To use this plugin with PipeWire as a system-wide noise suppression filter:

1.  **Install**: Copy the compiled `.so` file to your LADSPA path (e.g., `/usr/lib/ladspa/libgtcrn_ladspa.so`).
2.  **Config**: Create a file like `~/.config/pipewire/filter-chain.conf.d/gtcrn.conf`:

```conf
context.modules = [
    {
        name = libpipewire-module-filter-chain
        args = {
            node.description = "Noise Canceling Microphone (GTCRN)"
            media.name = "Noise Canceling Microphone (GTCRN)"
            filter.graph = {
                nodes = [
                    {
                        type = ladspa
                        name = "gtcrn"
                        plugin = "libgtcrn_ladspa"
                        label = "gtcrn_mono"
                        control = {
                            # Strength: 0.0 (Original) to 1.0 (Fully Processed)
                            "Strength" = 1.0
                            "Model (0=Light 1=Full)" = 1
                        }
                    }
                ]
            }
            audio.channels = 1
            capture.props = {
                node.passive = true
            }
            playback.props = {
                media.class = "Audio/Source"
            }
        }
    }
]
```

3.  **Run**: Start PipeWire with the filter chain enabled:
    ```bash
    pipewire -c filter-chain.conf
    ```

