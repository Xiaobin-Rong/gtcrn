//! GTCRN LADSPA Plugin using ONNX Runtime with OpenVINO backend.
//!
//! This plugin provides real-time speech enhancement using the GTCRN neural network,
//! with ONNX Runtime as the inference backend. It can use OpenVINO as execution provider
//! for optimal CPU performance on Intel hardware.

pub mod model;
pub mod plugin;
pub mod stft;

use ladspa::{
    DefaultValue, PluginDescriptor, Port, PortDescriptor, HINT_INTEGER, HINT_TOGGLED,
    PROP_HARD_REALTIME_CAPABLE, PROP_REALTIME,
};

/// Unique LADSPA plugin ID\
pub const PLUGIN_ID: u64 = 0x4F52_5443; // "ORTC" in ASCII hex

/// Plugin version string
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Port index for audio input
pub const PORT_INPUT: usize = 0;

/// Port index for audio output
pub const PORT_OUTPUT: usize = 1;

/// Port index for enable control
pub const PORT_ENABLE: usize = 2;

/// Port index for strength control
pub const PORT_STRENGTH: usize = 3;

/// Port index for model selection control
pub const PORT_MODEL: usize = 4;

/// Returns the LADSPA plugin descriptor.
#[no_mangle]
#[allow(unsafe_code)]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn get_ladspa_descriptor(index: u64) -> Option<PluginDescriptor> {
    if index != 0 {
        return None;
    }

    Some(PluginDescriptor {
        unique_id: PLUGIN_ID,
        label: "gtcrn_mono",
        properties: PROP_HARD_REALTIME_CAPABLE | PROP_REALTIME,
        name: "GTCRN Speech Enhancement (ORT)",
        maker: "GTCRN Model (c) 2024 Rong Xiaobin | Ladspa plugin (c) 2026 Bruno Gonçalves",
        copyright: "MIT License",
        ports: vec![
            Port {
                name: "Input",
                desc: PortDescriptor::AudioInput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            Port {
                name: "Output",
                desc: PortDescriptor::AudioOutput,
                hint: None,
                default: None,
                lower_bound: None,
                upper_bound: None,
            },
            Port {
                name: "Enable",
                desc: PortDescriptor::ControlInput,
                hint: Some(HINT_TOGGLED),
                default: Some(DefaultValue::Value1),
                lower_bound: None,
                upper_bound: None,
            },
            Port {
                name: "Strength",
                desc: PortDescriptor::ControlInput,
                hint: None,
                default: Some(DefaultValue::High),
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
            Port {
                name: "Model (0=Light 1=Full)",
                desc: PortDescriptor::ControlInput,
                hint: Some(HINT_INTEGER),
                default: Some(DefaultValue::Value1), // 1 = Full quality (default)
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
            },
        ],
        new: plugin::GtcrnPlugin::new,
    })
}

/// Re-export for LADSPA host discovery
pub use ladspa::ladspa_descriptor;
