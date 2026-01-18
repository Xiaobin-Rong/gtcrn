//! GTCRN neural network model wrapper using ONNX Runtime.
//!
//! This module provides a high-level interface to the GTCRN ONNX model,
//! using ONNX Runtime with optional OpenVINO execution provider.

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

// =============================================================================
// Embedded Models
// =============================================================================

/// Default embedded ONNX model (`gtcrn_simple.onnx`) - lighter, faster
// Load the converted ORT format model (required for minimal build)
static EMBEDDED_MODEL_SIMPLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gtcrn_simple.ort"));

/// Full quality embedded ONNX model (`gtcrn.onnx`)
/// Pre-optimized with ONNX Runtime to eliminate Range operators
static EMBEDDED_MODEL_FULL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gtcrn.ort"));

// =============================================================================
// Constants
// =============================================================================

/// Number of frequency bins in STFT output (NFFT/2 + 1 = 257)
pub const NUM_FREQ_BINS: usize = 257;

/// State array sizes (flat)
/// These correspond to tensor shapes used in GTCRN model:
/// - conv: (2, 1, 16, 16, 33) = 16896
/// - tra: (2, 3, 1, 1, 16) = 96
/// - inter: (2, 1, 33, 16) = 1056
/// - input: (1, 257, 1, 2) = 514
const CONV_SIZE: usize = 16896;
const TRA_SIZE: usize = 96;
const INTER_SIZE: usize = 1056;
const INPUT_SIZE: usize = NUM_FREQ_BINS * 2;

/// Model type selection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    /// Simple/fast model (`gtcrn_simple.onnx`)
    Simple = 0,
    /// Full quality model (`gtcrn.onnx`)
    Full = 1,
}

impl ModelType {
    /// Creates a `ModelType` from a control value (0=simple, 1=full)
    #[must_use]
    pub fn from_control(value: f32) -> Self {
        if value >= 0.5 {
            Self::Full
        } else {
            Self::Simple
        }
    }
}

// =============================================================================
// Model Loading
// =============================================================================

/// Creates a session from embedded model bytes
fn create_session(model_bytes: &[u8]) -> Result<Session, ort::Error> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .commit_from_memory(model_bytes)
}

// =============================================================================
// State
// =============================================================================

/// GTCRN recurrent state for streaming inference.
pub struct GtcrnState {
    /// Convolution state: shape (2, 1, 16, 16, 33) as flat f32
    conv: Vec<f32>,
    /// Temporal Recurrent Attention state: shape (2, 3, 1, 1, 16) as flat f32
    tra: Vec<f32>,
    /// Inter-frame DPGRNN state: shape (2, 1, 33, 16) as flat f32
    inter: Vec<f32>,
    /// Pre-allocated input buffer as flat f32: shape (1, 257, 1, 2)
    input_buf: Vec<f32>,
    /// Pre-allocated output buffer
    output_buf: [(f32, f32); NUM_FREQ_BINS],
}

impl GtcrnState {
    /// Creates a new zeroed state for streaming inference.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conv: vec![0.0_f32; CONV_SIZE],
            tra: vec![0.0_f32; TRA_SIZE],
            inter: vec![0.0_f32; INTER_SIZE],
            input_buf: vec![0.0_f32; INPUT_SIZE],
            output_buf: [(0.0_f32, 0.0_f32); NUM_FREQ_BINS],
        }
    }

    /// Resets the state to zeros.
    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.tra.fill(0.0);
        self.inter.fill(0.0);
    }
}

impl Default for GtcrnState {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for GtcrnState {
    fn clone(&self) -> Self {
        Self {
            conv: self.conv.clone(),
            tra: self.tra.clone(),
            inter: self.inter.clone(),
            input_buf: vec![0.0_f32; INPUT_SIZE],
            output_buf: [(0.0_f32, 0.0_f32); NUM_FREQ_BINS],
        }
    }
}

// =============================================================================
// Model Wrapper
// =============================================================================

/// GTCRN model wrapper for ONNX Runtime inference.
pub struct GtcrnModel {
    /// Current model type
    model_type: ModelType,
    /// Streaming state
    state: GtcrnState,
    /// ONNX Runtime session
    session: Session,
}

impl GtcrnModel {
    /// Creates a new model instance with specified type.
    #[must_use]
    pub fn new(model_type: ModelType) -> Self {
        eprintln!("GTCRN-ORT: Creating instance with {model_type:?} model");

        let model_bytes = match model_type {
            ModelType::Simple => EMBEDDED_MODEL_SIMPLE,
            ModelType::Full => EMBEDDED_MODEL_FULL,
        };

        let session = create_session(model_bytes).expect("Failed to create session");

        Self {
            model_type,
            state: GtcrnState::new(),
            session,
        }
    }

    /// Creates a new model instance using default (simple).
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(ModelType::Simple)
    }

    /// Gets the current model type.
    #[must_use]
    pub const fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Switches to a different model type.
    pub fn set_model_type(&mut self, model_type: ModelType) {
        if self.model_type != model_type {
            eprintln!("GTCRN-ORT: Switching to {model_type:?} model");

            let model_bytes = match model_type {
                ModelType::Simple => EMBEDDED_MODEL_SIMPLE,
                ModelType::Full => EMBEDDED_MODEL_FULL,
            };

            // Reload session
            match create_session(model_bytes) {
                Ok(new_session) => {
                    self.session = new_session;
                    self.model_type = model_type;
                    self.state.reset();
                }
                Err(e) => {
                    eprintln!("GTCRN-ORT: Failed to switch model session! Error: {e:?}");
                }
            }
        }
    }

    /// Resets the streaming state.
    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    /// Processes a single spectrum frame through the neural network.
    pub fn process_frame(
        &mut self,
        spectrum: &[(f32, f32); NUM_FREQ_BINS],
    ) -> Result<[(f32, f32); NUM_FREQ_BINS], Box<dyn std::error::Error + Send + Sync>> {
        // Fill input buffer (flat layout: [1, 257, 1, 2])
        for (i, &(re, im)) in spectrum.iter().enumerate() {
            self.state.input_buf[i * 2] = re;
            self.state.input_buf[i * 2 + 1] = im;
        }

        // Create tensor references using (shape, slice) tuples
        let input_tensor =
            TensorRef::from_array_view(([1usize, NUM_FREQ_BINS, 1, 2], &self.state.input_buf[..]))?;
        let conv_tensor =
            TensorRef::from_array_view(([2usize, 1, 16, 16, 33], &self.state.conv[..]))?;
        let tra_tensor = TensorRef::from_array_view(([2usize, 3, 1, 1, 16], &self.state.tra[..]))?;
        let inter_tensor =
            TensorRef::from_array_view(([2usize, 1, 33, 16], &self.state.inter[..]))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            input_tensor,
            conv_tensor,
            tra_tensor,
            inter_tensor,
        ])?;

        // Extract outputs using try_extract_tensor
        let (_, output_enh_data) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, output_conv_data) = outputs[1].try_extract_tensor::<f32>()?;
        let (_, output_tra_data) = outputs[2].try_extract_tensor::<f32>()?;
        let (_, output_inter_data) = outputs[3].try_extract_tensor::<f32>()?;

        // Update state
        self.state.conv.copy_from_slice(output_conv_data);
        self.state.tra.copy_from_slice(output_tra_data);
        self.state.inter.copy_from_slice(output_inter_data);

        // Extract enhanced spectrum
        for (i, pair) in self.state.output_buf.iter_mut().enumerate() {
            *pair = (output_enh_data[i * 2], output_enh_data[i * 2 + 1]);
        }

        Ok(self.state.output_buf)
    }
}
