//! GTCRN LADSPA Plugin implementation.
//!
//! This module implements the LADSPA plugin interface, handling:
//! - Audio I/O at host sample rate (typically 48kHz)
//! - High-quality resampling to model sample rate (16kHz) using sinc interpolation
//! - Lock-free ring buffer communication for real-time audio
//! - Adjustable enhancement strength for blending original/processed audio
//! - Model selection via control port
//!
//! ## Architecture
//!
//! ```text
//! Audio Thread (real-time)          Worker Thread (non-real-time)
//! ┌─────────────────────────┐       ┌─────────────────────────────┐
//! │ run() callback          │       │ worker_thread()             │
//! │   ├─ input → input_ring │──────>│   ├─ downsample 48k→16k    │
//! │   └─ output_ring → out  │<──────│   ├─ STFT → Model → iSTFT  │
//! └─────────────────────────┘       │   └─ upsample 16k→48k      │
//!                                   └─────────────────────────────┘
//! ```

use crate::model::{GtcrnModel, ModelType, NUM_FREQ_BINS};
use crate::stft::{StftProcessor, HOP_SIZE, NFFT};
use crate::{PORT_ENABLE, PORT_INPUT, PORT_MODEL, PORT_OUTPUT, PORT_STRENGTH};
use ladspa::{Plugin, PluginDescriptor, PortConnection};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapRb,
};
use rubato::{FftFixedIn, FftFixedOut, Resampler};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

// =============================================================================
// Constants
// =============================================================================

/// Sample rate expected by the GTCRN model
const MODEL_SAMPLE_RATE: usize = 16_000;

/// Output gain to compensate for model's natural volume reduction
const OUTPUT_GAIN: f32 = 1.45;

/// Minimum sleep duration for worker thread polling (microseconds)
const WORKER_SLEEP_MIN_US: u64 = 500;

/// Maximum sleep duration for worker thread polling (microseconds)
/// ~5ms matches typical audio buffer sizes
const WORKER_SLEEP_MAX_US: u64 = 5000;

/// Ring buffer capacity in samples (covers multiple audio blocks)
/// At 48kHz with 1024 sample blocks, this covers ~170ms
const RING_BUFFER_SIZE: usize = 8192;

// =============================================================================
// Shared State
// =============================================================================

/// Shared state between audio thread and worker thread.
/// Uses atomic operations for lock-free communication.
struct SharedState {
    /// Processing enabled flag
    enabled: AtomicBool,
    /// Strength value as u32 bits (for atomic f32 storage)
    strength_bits: AtomicU32,
    /// Model type value as u32 bits
    model_bits: AtomicU32,
    /// Shutdown signal for clean termination
    shutdown: AtomicBool,
    /// Flag indicating port values have been read from first run() call
    initialized: AtomicBool,
}

impl SharedState {
    fn new(model_type: ModelType) -> Self {
        Self {
            enabled: AtomicBool::new(true),
            strength_bits: AtomicU32::new(1.0_f32.to_bits()),
            model_bits: AtomicU32::new((model_type as u32 as f32).to_bits()),
            shutdown: AtomicBool::new(false),
            initialized: AtomicBool::new(false),
        }
    }

    #[inline]
    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    #[inline]
    fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    #[inline]
    fn set_strength(&self, strength: f32) {
        let clamped = strength.clamp(0.0, 1.0);
        self.strength_bits
            .store(clamped.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn get_strength(&self) -> f32 {
        f32::from_bits(self.strength_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_model(&self, value: f32) {
        self.model_bits.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn get_model_type(&self) -> ModelType {
        let value = f32::from_bits(self.model_bits.load(Ordering::Relaxed));
        ModelType::from_control(value)
    }

    #[inline]
    fn should_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    #[inline]
    fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    #[inline]
    fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    #[inline]
    fn set_initialized(&self) {
        self.initialized.store(true, Ordering::Release);
    }
}

// =============================================================================
// Worker Thread
// =============================================================================

/// Worker thread for audio processing.
///
/// Processes audio in a separate thread to avoid blocking the real-time
/// audio callback. Uses pre-allocated buffers to prevent memory allocations
/// during processing. Model is created based on the initial state value.
fn worker_thread(
    mut input_consumer: ringbuf::HeapCons<f32>,
    mut output_producer: ringbuf::HeapProd<f32>,
    state: Arc<SharedState>,
    host_sample_rate: usize,
) {
    let needs_resample = host_sample_rate != MODEL_SAMPLE_RATE;
    let resample_ratio = host_sample_rate as f64 / MODEL_SAMPLE_RATE as f64;

    // Calculate chunk sizes for resampling
    // At 48kHz host and 16kHz model, ratio is 3.0
    // We process HOP_SIZE (256) samples at model rate per frame
    let host_chunk_size = (HOP_SIZE as f64 * resample_ratio).ceil() as usize;

    // Initialize high-quality sinc resamplers
    let mut downsampler: Option<FftFixedIn<f32>> = if needs_resample {
        Some(
            FftFixedIn::new(host_sample_rate, MODEL_SAMPLE_RATE, host_chunk_size, 1, 1)
                .expect("Failed to create downsampler"),
        )
    } else {
        None
    };

    let mut upsampler: Option<FftFixedOut<f32>> = if needs_resample {
        Some(
            FftFixedOut::new(MODEL_SAMPLE_RATE, host_sample_rate, host_chunk_size, 1, 1)
                .expect("Failed to create upsampler"),
        )
    } else {
        None
    };

    // Initialize STFT processor
    let mut stft = StftProcessor::new(NFFT, HOP_SIZE);

    // Wait for first run() call to set port values before creating model
    while !state.is_initialized() {
        if state.should_shutdown() {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }

    // Create model based on current state value (now set by run() call)
    let initial_model_type = state.get_model_type();
    let mut model = GtcrnModel::new(initial_model_type);

    // Pre-allocated fixed-size buffers (no heap allocations during processing)
    let mut input_buffer = vec![0.0_f32; host_chunk_size + 64];
    let mut window = vec![0.0_f32; NFFT];
    let mut model_accum = vec![0.0_f32; HOP_SIZE * 4]; // Accumulator for downsampled samples
    let mut model_accum_len: usize = 0;

    // Resampler I/O buffers (single channel, fixed size)
    let mut resample_in = vec![vec![0.0_f32; host_chunk_size + 64]];
    let mut resample_out = vec![vec![0.0_f32; HOP_SIZE + 64]];
    let mut upsample_in = vec![vec![0.0_f32; HOP_SIZE + 64]];
    let mut upsample_out = vec![vec![0.0_f32; host_chunk_size + 64]];

    // Output accumulator for upsampled data
    let mut output_accum = vec![0.0_f32; host_chunk_size * 4];
    let mut output_accum_len: usize = 0;

    // Adaptive backoff: start with minimum sleep, increase when idle
    let mut current_sleep_us = WORKER_SLEEP_MIN_US;

    // Pre-computed spectrum buffer to avoid allocation in hot path
    let mut spectrum_buffer: [(f32, f32); NUM_FREQ_BINS] = [(0.0, 0.0); NUM_FREQ_BINS];

    loop {
        if state.should_shutdown() {
            break;
        }

        // Calculate required samples
        let required_samples = if needs_resample {
            downsampler.as_ref().unwrap().input_frames_next()
        } else {
            HOP_SIZE
        };

        // Check available samples (lock-free)
        let available = input_consumer.occupied_len();
        if available < required_samples {
            // No data available - sleep with adaptive backoff
            thread::sleep(Duration::from_micros(current_sleep_us));
            // Increase sleep duration for next iteration (up to max)
            current_sleep_us = (current_sleep_us * 2).min(WORKER_SLEEP_MAX_US);
            continue;
        }

        // Data available - reset backoff to minimum
        current_sleep_us = WORKER_SLEEP_MIN_US;

        // Read samples from ring buffer (lock-free)
        let samples_read = input_consumer.pop_slice(&mut input_buffer[..required_samples]);
        if samples_read < required_samples {
            // Race condition: buffer drained between check and read
            thread::sleep(Duration::from_micros(WORKER_SLEEP_MIN_US));
            continue;
        }

        // Get control values (atomic reads)
        let is_enabled = state.is_enabled();
        let strength = state.get_strength();

        // Update model type if changed
        let requested_model = state.get_model_type();
        if model.model_type() != requested_model {
            model.set_model_type(requested_model);
        }

        // Bypass mode: pass through input directly
        if !is_enabled {
            // Write input directly to output (with resampling if needed to maintain sync)
            let written = output_producer.push_slice(&input_buffer[..samples_read]);
            if written < samples_read {
                // Output buffer full - drop samples to prevent buildup
            }
            continue;
        }

        // Downsample to 16kHz if needed
        if needs_resample {
            // Copy to resampler input buffer
            resample_in[0][..samples_read].copy_from_slice(&input_buffer[..samples_read]);

            let ds = downsampler.as_mut().unwrap();
            let frames_needed = ds.output_frames_next();
            resample_out[0].resize(frames_needed, 0.0);

            match ds.process_into_buffer(&resample_in, &mut resample_out, None) {
                Ok((_, out_frames)) => {
                    // Copy to model accumulator
                    let space_available = model_accum.len() - model_accum_len;
                    let to_copy = out_frames.min(space_available);
                    model_accum[model_accum_len..model_accum_len + to_copy]
                        .copy_from_slice(&resample_out[0][..to_copy]);
                    model_accum_len += to_copy;
                }
                Err(_) => {
                    // On error, use input directly (wrong rate but prevents silence)
                    let to_copy = samples_read.min(model_accum.len() - model_accum_len);
                    model_accum[model_accum_len..model_accum_len + to_copy]
                        .copy_from_slice(&input_buffer[..to_copy]);
                    model_accum_len += to_copy;
                }
            }
        } else {
            // No resampling needed
            let to_copy = samples_read.min(model_accum.len() - model_accum_len);
            model_accum[model_accum_len..model_accum_len + to_copy]
                .copy_from_slice(&input_buffer[..to_copy]);
            model_accum_len += to_copy;
        }

        // Process complete frames through STFT -> Model -> iSTFT
        while model_accum_len >= HOP_SIZE {
            // Shift window and add new samples
            window.copy_within(HOP_SIZE.., 0);
            window[NFFT - HOP_SIZE..].copy_from_slice(&model_accum[..HOP_SIZE]);

            // Remove used samples from accumulator
            model_accum.copy_within(HOP_SIZE..model_accum_len, 0);
            model_accum_len -= HOP_SIZE;

            // Process through STFT -> Model -> iSTFT
            let spectrum = stft.analyze(&window);
            spectrum_buffer.copy_from_slice(spectrum);

            let enhanced = model
                .process_frame(&spectrum_buffer)
                .unwrap_or(spectrum_buffer);
            let processed = stft.synthesize(&enhanced);

            // Upsample back to host rate if needed
            if needs_resample {
                // Copy to upsampler input
                upsample_in[0][..processed.len()].copy_from_slice(processed);

                let us = upsampler.as_mut().unwrap();
                let frames_needed = us.output_frames_next();
                upsample_out[0].resize(frames_needed, 0.0);

                match us.process_into_buffer(&upsample_in, &mut upsample_out, None) {
                    Ok((_, out_frames)) => {
                        // Apply gain and add to output accumulator
                        // Branch moved outside loop for better performance
                        let space_available = output_accum.len() - output_accum_len;
                        let to_copy = out_frames.min(space_available);
                        let gain = OUTPUT_GAIN * strength;
                        for i in 0..to_copy {
                            output_accum[output_accum_len + i] = upsample_out[0][i] * gain;
                        }
                        output_accum_len += to_copy;
                    }
                    Err(_) => {
                        // Fallback: simple upsampling by repetition
                        let space_available = output_accum.len() - output_accum_len;
                        let to_copy = (processed.len() * 3).min(space_available);
                        let gain = OUTPUT_GAIN * strength;
                        for i in 0..to_copy {
                            output_accum[output_accum_len + i] = processed[i / 3] * gain;
                        }
                        output_accum_len += to_copy;
                    }
                }
            } else {
                // No upsampling - copy directly with gain
                let space_available = output_accum.len() - output_accum_len;
                let to_copy = processed.len().min(space_available);
                let gain = OUTPUT_GAIN * strength;
                for i in 0..to_copy {
                    output_accum[output_accum_len + i] = processed[i] * gain;
                }
                output_accum_len += to_copy;
            }
        }

        // Write accumulated output to ring buffer (lock-free)
        if output_accum_len > 0 {
            let written = output_producer.push_slice(&output_accum[..output_accum_len]);
            if written < output_accum_len {
                // Partial write - keep remaining samples
                output_accum.copy_within(written..output_accum_len, 0);
                output_accum_len -= written;
            } else {
                output_accum_len = 0;
            }
        }
    }
}

// =============================================================================
// Plugin Implementation
// =============================================================================

/// GTCRN LADSPA plugin instance.
pub struct GtcrnPlugin {
    /// Input ring buffer producer
    input_producer: ringbuf::HeapProd<f32>,
    /// Output ring buffer consumer
    output_consumer: ringbuf::HeapCons<f32>,
    /// Worker thread handle
    worker: Option<JoinHandle<()>>,
    /// Shared state
    state: Arc<SharedState>,
    /// Host sample rate
    #[allow(dead_code)]
    host_sample_rate: usize,
}

impl GtcrnPlugin {
    /// Creates a new plugin instance.
    ///
    /// Returns a boxed trait object as required by the LADSPA plugin interface.
    #[must_use]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(_descriptor: &PluginDescriptor, sample_rate: u64) -> Box<dyn Plugin + Send> {
        let host_sr = sample_rate as usize;

        // Create lock-free ring buffers
        let input_ring = HeapRb::<f32>::new(RING_BUFFER_SIZE);
        let output_ring = HeapRb::<f32>::new(RING_BUFFER_SIZE);

        let (input_producer, input_consumer) = input_ring.split();
        let (mut output_producer, output_consumer) = output_ring.split();

        // Pre-fill output buffer for latency compensation
        // Two hops worth at host rate for smoother startup
        let resample_ratio = host_sr as f64 / MODEL_SAMPLE_RATE as f64;
        let prefill_samples = ((HOP_SIZE as f64 * resample_ratio) * 2.0) as usize;
        let zeros = vec![0.0_f32; prefill_samples];
        output_producer.push_slice(&zeros);

        // Use Simple as default - will be updated when run() is called with port values
        let state = Arc::new(SharedState::new(ModelType::Simple));

        // Spawn worker thread - model will be created inside based on state
        let worker = {
            let st = Arc::clone(&state);
            Some(thread::spawn(move || {
                worker_thread(input_consumer, output_producer, st, host_sr)
            }))
        };

        Box::new(Self {
            input_producer,
            output_consumer,
            worker,
            state,
            host_sample_rate: host_sr,
        })
    }
}

impl Drop for GtcrnPlugin {
    fn drop(&mut self) {
        // Signal worker to shutdown
        self.state.request_shutdown();

        // Wait for worker to finish
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

impl Plugin for GtcrnPlugin {
    fn activate(&mut self) {
        // Note: Ring buffers are not easily cleared without recreation.
        // The prefill ensures we start with known state.
    }

    fn deactivate(&mut self) {
        // Nothing needed - worker continues running
    }

    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let input = ports[PORT_INPUT].unwrap_audio();
        let mut output = ports[PORT_OUTPUT].unwrap_audio_mut();
        let enable_control = *ports[PORT_ENABLE].unwrap_control();
        let strength_control = *ports[PORT_STRENGTH].unwrap_control();
        let model_control = *ports[PORT_MODEL].unwrap_control();

        // Update shared state (atomic writes - no locks)
        self.state.set_enabled(enable_control >= 0.5);
        self.state.set_strength(strength_control);
        self.state.set_model(model_control);

        // Signal that port values are now available (enables model creation in worker)
        if !self.state.is_initialized() {
            self.state.set_initialized();
        }

        // Push input samples to ring buffer (lock-free)
        let written = self.input_producer.push_slice(&input[..sample_count]);
        if written < sample_count {
            // Ring buffer full - discard oldest samples wouldn't help here
            // Just proceed with partial write
        }

        // Get output samples (lock-free)
        let read = self.output_consumer.pop_slice(&mut output[..sample_count]);

        // Fill any remaining output with passthrough input
        if read < sample_count {
            for i in read..sample_count {
                output[i] = input[i] * 0.95; // Slight attenuation to indicate we're behind
            }
        }
    }
}
