//! Real-time STFT/iSTFT processing with overlap-add synthesis.
//!
//! Implements Short-Time Fourier Transform and its inverse using the
//! `sqrt(hann)` window for perfect reconstruction with 50% overlap.
//!
//! This implementation is optimized for real-time audio processing:
//! - Minimal heap allocations (only during initialization)
//! - Returns slices to internal buffers to avoid copies
//! - Pre-allocated scratch buffers

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

use crate::model::NUM_FREQ_BINS;

/// Hop size in samples (256 for GTCRN with 50% overlap)
pub const HOP_SIZE: usize = 256;

/// FFT size (512 for GTCRN)
pub const NFFT: usize = 512;

/// STFT processor for real-time streaming audio.
///
/// Uses 512-point FFT with 256-sample hop (50% overlap) as required by GTCRN.
/// The `sqrt(hann)` window ensures perfect reconstruction when applied to
/// both analysis and synthesis stages.
///
/// # Performance
///
/// This processor minimizes allocations by reusing internal buffers.
/// Methods return slices to internal buffers rather than copying data.
pub struct StftProcessor {
    /// FFT size (512 for GTCRN)
    nfft: usize,
    /// Hop size in samples (256 for GTCRN)
    hop_size: usize,
    /// Analysis/synthesis window: sqrt(hann)
    window: Vec<f32>,
    /// Forward FFT plan
    fft: Arc<dyn RealToComplex<f32>>,
    /// Inverse FFT plan
    ifft: Arc<dyn ComplexToReal<f32>>,
    /// Overlap-add buffer for synthesis
    overlap_buffer: Vec<f32>,
    /// Scratch buffer for FFT input
    fft_scratch: Vec<f32>,
    /// Scratch buffer for FFT output
    spectrum_scratch: Vec<Complex<f32>>,
    /// Pre-allocated iFFT output buffer
    ifft_scratch: Vec<f32>,
    /// Pre-allocated output spectrum buffer (avoids allocation in analyze)
    output_spectrum: [(f32, f32); NUM_FREQ_BINS],
    /// Pre-allocated output samples buffer (avoids allocation in synthesize)
    output_samples: Vec<f32>,
}

impl StftProcessor {
    /// Creates a new STFT processor with given parameters.
    ///
    /// # Arguments
    ///
    /// * `nfft` - FFT size (should be 512 for GTCRN)
    /// * `hop_size` - Hop size in samples (should be 256 for GTCRN)
    #[must_use]
    pub fn new(nfft: usize, hop_size: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);
        let ifft = planner.plan_fft_inverse(nfft);

        // sqrt(Hann) window - periodic style matching PyTorch's default
        let window: Vec<f32> = (0..nfft)
            .map(|i| {
                let phase = 2.0 * PI * i as f32 / nfft as f32;
                let hann = 0.5 * (1.0 - phase.cos());
                hann.sqrt()
            })
            .collect();

        Self {
            nfft,
            hop_size,
            window,
            fft,
            ifft,
            overlap_buffer: vec![0.0; nfft],
            fft_scratch: vec![0.0; nfft],
            spectrum_scratch: vec![Complex::new(0.0, 0.0); nfft / 2 + 1],
            ifft_scratch: vec![0.0; nfft],
            output_spectrum: [(0.0, 0.0); NUM_FREQ_BINS],
            output_samples: vec![0.0; hop_size],
        }
    }

    /// Analyzes a time-domain frame and returns its frequency spectrum.
    ///
    /// # Arguments
    ///
    /// * `frame` - Time-domain samples (must be exactly `nfft` samples)
    ///
    /// # Returns
    ///
    /// Reference to internal spectrum buffer as (real, imag) pairs.
    /// Valid until next call to `analyze`.
    ///
    /// # Panics
    ///
    /// Panics if FFT processing fails (should never happen with valid input).
    #[inline]
    pub fn analyze(&mut self, frame: &[f32]) -> &[(f32, f32); NUM_FREQ_BINS] {
        // Apply analysis window
        for (i, (&sample, &win)) in frame.iter().zip(&self.window).enumerate() {
            self.fft_scratch[i] = sample * win;
        }

        // Forward FFT
        self.fft
            .process(&mut self.fft_scratch, &mut self.spectrum_scratch)
            .expect("FFT processing failed");

        // Convert to (real, imag) tuple format into pre-allocated buffer
        for (i, c) in self.spectrum_scratch.iter().enumerate().take(NUM_FREQ_BINS) {
            self.output_spectrum[i] = (c.re, c.im);
        }

        &self.output_spectrum
    }

    /// Synthesizes time-domain samples from a frequency spectrum.
    ///
    /// Uses overlap-add for smooth reconstruction across frames.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Complex spectrum as (real, imag) pairs
    ///
    /// # Returns
    ///
    /// Slice of time-domain samples (`hop_size` samples).
    /// Valid until next call to `synthesize`.
    #[inline]
    pub fn synthesize(&mut self, spectrum: &[(f32, f32); NUM_FREQ_BINS]) -> &[f32] {
        // Convert (real, imag) tuples to Complex
        for (i, &(re, im)) in spectrum.iter().enumerate() {
            self.spectrum_scratch[i] = Complex::new(re, im);
        }

        // DC and Nyquist bins must have zero imaginary part
        self.spectrum_scratch[0].im = 0.0;
        self.spectrum_scratch[NUM_FREQ_BINS - 1].im = 0.0;

        // Inverse FFT into pre-allocated buffer
        if self
            .ifft
            .process(&mut self.spectrum_scratch, &mut self.ifft_scratch)
            .is_err()
        {
            self.output_samples.fill(0.0);
            return &self.output_samples;
        }

        // Normalize iFFT and apply synthesis window
        let scale = 1.0 / self.nfft as f32;
        for (i, sample) in self.ifft_scratch.iter_mut().enumerate() {
            *sample *= scale * self.window[i];
        }

        // Overlap-add accumulation
        for (i, &sample) in self.ifft_scratch.iter().enumerate() {
            self.overlap_buffer[i] += sample;
        }

        // Copy output samples to pre-allocated buffer
        self.output_samples
            .copy_from_slice(&self.overlap_buffer[..self.hop_size]);

        // Shift buffer for next frame
        self.overlap_buffer.copy_within(self.hop_size.., 0);
        self.overlap_buffer[self.nfft - self.hop_size..].fill(0.0);

        &self.output_samples
    }
}
