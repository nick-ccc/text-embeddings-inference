use std::collections::HashMap;

use crate::layers::{
    apply_rotary, get_cublas_lt_wrapper, get_inv_freqs, HiddenAct, Linear, RopeParameters,
};
use crate::models::Model;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingGemma2LayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2LayerConfig {
    pub head_dim: usize,
    pub num_key_value_heads: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2TextConfig {
    pub attention_bias: bool,
    pub pad_token_id: u32,
    pub head_dim: usize,
    pub hidden_activation: HiddenAct,
    pub hidden_size: usize,
    pub hidden_size_per_layer_input: usize,
    pub embedding_dim: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: HashMap<EmbeddingGemma2LayerType, RopeParameters>,
    pub sliding_window: usize,
    pub layer_types: Vec<EmbeddingGemma2LayerType>,
    // NOTE: Keyed by the layer index as a string, e.g. `"05"`, overriding `head_dim` and
    // `num_key_value_heads` for that layer (the full attention layers are wider)
    #[serde(default)]
    pub per_layer_config: HashMap<String, EmbeddingGemma2LayerConfig>,
    pub vocab_size: usize,
}

impl EmbeddingGemma2TextConfig {
    fn layer_config(&self, layer_idx: usize) -> EmbeddingGemma2LayerConfig {
        self.per_layer_config
            .iter()
            .find(|(key, _)| key.parse::<usize>().ok() == Some(layer_idx))
            .map(|(_, layer_config)| layer_config.clone())
            .unwrap_or(EmbeddingGemma2LayerConfig {
                head_dim: self.head_dim,
                num_key_value_heads: self.num_key_value_heads,
            })
    }
}

/// Only the text tower is supported, so the `vision_config` and `audio_config` are ignored, as well
/// as the `vision_tower`, `embed_vision`, `audio_tower` and `embed_audio` weights
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmbeddingGemma2Config {
    pub text_config: EmbeddingGemma2TextConfig,
}

/// Unlike Gemma3, the RMSNorm scales by `weight` rather than by `1.0 + weight`, and the `v_norm`
/// comes with no weight at all
#[derive(Debug)]
struct EmbeddingGemma2RMSNorm {
    weight: Tensor,
    epsilon: f32,

    span: tracing::Span,
}

impl EmbeddingGemma2RMSNorm {
    pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: vb.get(hidden_size, "weight")?,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        })
    }

    pub fn load_without_scale(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
        Ok(Self {
            weight: Tensor::ones(hidden_size, vb.dtype(), vb.device())?,
            epsilon,
            span: tracing::span!(tracing::Level::TRACE, "rms-norm"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        match hidden_states.device() {
            Device::Cpu | Device::Metal(_) => {
                let hidden_states_dtype = hidden_states.dtype();
                let internal_dtype = match hidden_states_dtype {
                    DType::F16 | DType::BF16 => DType::F32,
                    d => d,
                };

                let hidden_size = hidden_states.dim(D::Minus1)?;
                let hidden_states = hidden_states.to_dtype(internal_dtype)?;
                let norm_hidden_states =
                    (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
                let hidden_states_normed = hidden_states
                    .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?;
                hidden_states_normed
                    .to_dtype(hidden_states_dtype)?
                    .broadcast_mul(&self.weight)
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    use candle_layer_norm::rms_norm;

                    let original_shape = hidden_states.shape();
                    let hidden_states = hidden_states.flatten_to(D::Minus2)?;
                    let result = rms_norm(&hidden_states, &self.weight, None, self.epsilon)?;
                    result.reshape(original_shape)
                }
                #[cfg(not(feature = "cuda"))]
                candle::bail!("`cuda` feature is not enabled")
            }
        }
    }
}

struct EmbeddingGemma2Attention {
    qkv_proj: Linear,
    o_proj: Linear,

    q_norm: EmbeddingGemma2RMSNorm,
    k_norm: EmbeddingGemma2RMSNorm,
    v_norm: EmbeddingGemma2RMSNorm,

    attention_head_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2Attention {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let layer_config = config.layer_config(layer_idx);

        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = layer_config.head_dim;
        let num_key_value_heads = layer_config.num_key_value_heads;
        let hidden_size = config.hidden_size;

        let query_weight = vb.pp("q_proj").get(
            (num_attention_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let key_weight = vb.pp("k_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let value_weight = vb.pp("v_proj").get(
            (num_key_value_heads * attention_head_size, hidden_size),
            "weight",
        )?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;

        let qkv_bias = if config.attention_bias {
            let query_bias = vb
                .pp("q_proj")
                .get(num_attention_heads * attention_head_size, "bias")?;
            let key_bias = vb
                .pp("k_proj")
                .get(num_key_value_heads * attention_head_size, "bias")?;
            let value_bias = vb
                .pp("v_proj")
                .get(num_key_value_heads * attention_head_size, "bias")?;
            Some(Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?)
        } else {
            None
        };

        let qkv_proj = Linear::new(qkv_weight, qkv_bias, None);

        let output_weight = vb.pp("o_proj").get(
            (hidden_size, num_attention_heads * attention_head_size),
            "weight",
        )?;
        let output_bias = if config.attention_bias {
            Some(vb.pp("o_proj").get(hidden_size, "bias")?)
        } else {
            None
        };
        let o_proj = Linear::new(output_weight, output_bias, None);

        let q_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("q_norm"),
            attention_head_size,
            config.rms_norm_eps,
        )?;
        let k_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("k_norm"),
            attention_head_size,
            config.rms_norm_eps,
        )?;
        let v_norm = EmbeddingGemma2RMSNorm::load_without_scale(
            vb.pp("v_norm"),
            attention_head_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm,
            attention_head_size,
            num_attention_heads,
            num_key_value_heads,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let device = hidden_states.device();

        let qkv = self.qkv_proj.forward(hidden_states)?;

        let input_dims = hidden_states.dims();
        let input_shape = &input_dims[..input_dims.len() - 1];

        let q_size = self.num_attention_heads * self.attention_head_size;
        let k_size = self.num_key_value_heads * self.attention_head_size;
        let v_size = self.num_key_value_heads * self.attention_head_size;

        let q = qkv.narrow(D::Minus1, 0, q_size)?;
        let k = qkv.narrow(D::Minus1, q_size, k_size)?;
        let v = qkv.narrow(D::Minus1, q_size + k_size, v_size)?;

        let q = q.reshape(
            [
                input_shape,
                &[self.num_attention_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let k = k.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;
        let v = v.reshape(
            [
                input_shape,
                &[self.num_key_value_heads, self.attention_head_size],
            ]
            .concat(),
        )?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_norm.forward(&v)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = apply_rotary(&q, cos, sin, self.attention_head_size)?;
        let k = apply_rotary(&k, cos, sin, self.attention_head_size)?;

        // For simplicity, expand k and v to match number of q heads if needed (GQA)
        let k = if self.num_key_value_heads != self.num_attention_heads {
            let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
            let (b, h, s, d) = k.shape().dims4()?;
            let k = k.unsqueeze(2)?.expand((b, h, repeat_factor, s, d))?;
            k.reshape((b, h * repeat_factor, s, d))?
        } else {
            k
        };

        let v = if self.num_key_value_heads != self.num_attention_heads {
            let repeat_factor = self.num_attention_heads / self.num_key_value_heads;
            let (b, h, s, d) = v.shape().dims4()?;
            let v = v.unsqueeze(2)?.expand((b, h, repeat_factor, s, d))?;
            v.reshape((b, h * repeat_factor, s, d))?
        } else {
            v
        };

        // NOTE: The attention scaling is 1.0, given that the queries and keys are already normalized
        #[allow(unused_variables)]
        let context_layer = if let (Device::Cuda(_), Some(cublaslt)) =
            (device, get_cublas_lt_wrapper())
        {
            #[cfg(feature = "cuda")]
            {
                let (batch_size, _, seq_len, _) = k.shape().dims4()?;
                let q = q.flatten(0, 1)?;
                let k = k.flatten(0, 1)?;
                let v = v.flatten(0, 1)?;
                let attention_bias = attention_bias.map(|mask| mask.flatten(0, 1)).transpose()?;

                let beta = match attention_bias.is_some() {
                    true => Some(1.0),
                    false => None,
                };

                let attention_scores = cublaslt.batch_matmul(
                    &k,
                    &q,
                    attention_bias.as_ref(),
                    Some(1.0),
                    beta,
                    None,
                    None,
                )?;
                let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

                let context_layer = cublaslt.batch_matmul(
                    &v.t()?.contiguous()?,
                    &attention_probs,
                    Some(&q),
                    None,
                    None,
                    None,
                    None,
                )?;

                context_layer.reshape((
                    batch_size,
                    self.num_attention_heads,
                    seq_len,
                    self.attention_head_size,
                ))
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle::bail!("`cuda` feature is not enabled")
            }
        } else {
            let mut attn_weights = q.matmul(&k.t()?)?;

            if let Some(attention_bias) = attention_bias {
                attn_weights = attn_weights.broadcast_add(attention_bias)?;
            }

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v.contiguous()?)
        }?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;
        self.o_proj.forward(&context_layer)
    }
}

struct EmbeddingGemma2MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    hidden_activation: HiddenAct,

    intermediate_size: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2MLP {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let gate_proj_weight = vb
            .pp("gate_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;

        let up_proj_weight = vb
            .pp("up_proj")
            .get((config.intermediate_size, config.hidden_size), "weight")?;

        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None, None);

        let down_proj_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            hidden_activation: config.hidden_activation.clone(),
            intermediate_size: config.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;

        let gate_states = gate_up_states.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gate_states = self.hidden_activation.forward(&gate_states)?;

        let up_states =
            gate_up_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;

        self.down_proj.forward(&(gate_states * up_states)?)
    }
}

/// Computes the per-layer embeddings (PLE) from the input embeddings alone, as EmbeddingGemma2
/// uses projection-only PLE i.e. there's no per-layer token embedding table
struct EmbeddingGemma2PLE {
    per_layer_model_projection: Linear,
    per_layer_projection_norm: EmbeddingGemma2RMSNorm,
    scale: f64,

    num_hidden_layers: usize,
    hidden_size_per_layer_input: usize,

    span: tracing::Span,
}

impl EmbeddingGemma2PLE {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let per_layer_model_projection_weight = vb.pp("per_layer_model_projection").get(
            (
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                config.hidden_size,
            ),
            "weight",
        )?;
        let per_layer_model_projection = Linear::new(per_layer_model_projection_weight, None, None);

        let per_layer_projection_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("per_layer_projection_norm"),
            config.hidden_size_per_layer_input,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            per_layer_model_projection,
            per_layer_projection_norm,
            scale: (config.hidden_size as f64).powf(-0.5),
            num_hidden_layers: config.num_hidden_layers,
            hidden_size_per_layer_input: config.hidden_size_per_layer_input,
            span: tracing::span!(tracing::Level::TRACE, "ple"),
        })
    }

    pub fn forward(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let (batch_size, seq_len, _) = inputs_embeds.dims3()?;
        let per_layer_projection =
            (self.per_layer_model_projection.forward(inputs_embeds)? * self.scale)?;
        let per_layer_projection = per_layer_projection.reshape((
            batch_size,
            seq_len,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        ))?;
        self.per_layer_projection_norm
            .forward(&per_layer_projection)
    }
}

/// Gates the per-layer embeddings into the residual stream, after the attention and the MLP
struct EmbeddingGemma2PLEBlock {
    per_layer_input_gate: Linear,
    per_layer_projection: Linear,
    post_per_layer_input_norm: EmbeddingGemma2RMSNorm,
    hidden_activation: HiddenAct,

    span: tracing::Span,
}

impl EmbeddingGemma2PLEBlock {
    pub fn load(vb: VarBuilder, config: &EmbeddingGemma2TextConfig) -> Result<Self> {
        let per_layer_input_gate_weight = vb.pp("per_layer_input_gate").get(
            (config.hidden_size_per_layer_input, config.hidden_size),
            "weight",
        )?;
        let per_layer_input_gate = Linear::new(per_layer_input_gate_weight, None, None);

        let per_layer_projection_weight = vb.pp("per_layer_projection").get(
            (config.hidden_size, config.hidden_size_per_layer_input),
            "weight",
        )?;
        let per_layer_projection = Linear::new(per_layer_projection_weight, None, None);

        let post_per_layer_input_norm = EmbeddingGemma2RMSNorm::load(
            vb.pp("post_per_layer_input_norm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            hidden_activation: config.hidden_activation.clone(),
            span: tracing::span!(tracing::Level::TRACE, "ple_block"),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states;
        let hidden_states = self.per_layer_input_gate.forward(hidden_states)?;
        let hidden_states = self.hidden_activation.forward(&hidden_states)?;
        let hidden_states = hidden_states.broadcast_mul(per_layer_input)?;
        let hidden_states = self.per_layer_projection.forward(&hidden_states)?;
        let hidden_states = self.post_per_layer_input_norm.forward(&hidden_states)?;
        residual.broadcast_add(&hidden_states)
    }
}

struct EmbeddingGemma2Layer {
    input_layernorm: EmbeddingGemma2RMSNorm,
    self_attn: EmbeddingGemma2Attention,
    post_attention_layernorm: EmbeddingGemma2RMSNorm,

    pre_feedforward_layernorm: EmbeddingGemma2RMSNorm,
    mlp: EmbeddingGemma2MLP,
    post_feedforward_layernorm: EmbeddingGemma2RMSNorm,

    ple_block: EmbeddingGemma2PLEBlock,
    layer_scalar: Tensor,

    layer_type: EmbeddingGemma2LayerType,

    span: tracing::Span,
}

impl EmbeddingGemma2Layer {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2TextConfig,
        layer_idx: usize,
    ) -> Result<Self> {
        let input_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let self_attn = EmbeddingGemma2Attention::load(vb.pp("self_attn"), config, layer_idx)?;
        let post_attention_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let pre_feedforward_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("pre_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let mlp = EmbeddingGemma2MLP::load(vb.pp("mlp"), config)?;
        let post_feedforward_layernorm = EmbeddingGemma2RMSNorm::load(
            vb.pp("post_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let ple_block = EmbeddingGemma2PLEBlock::load(vb.pp("ple_block"), config)?;
        let layer_scalar = vb.get(1, "layer_scalar")?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            mlp,
            post_feedforward_layernorm,
            ple_block,
            layer_scalar,
            layer_type: config.layer_types[layer_idx],
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        per_layer_input: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

        let residual = hidden_states.clone();

        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, attention_bias, cos, sin)?;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = residual.broadcast_add(&hidden_states)?;

        let residual = hidden_states.clone();
        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = residual.broadcast_add(&hidden_states)?;

        let hidden_states = self.ple_block.forward(&hidden_states, per_layer_input)?;
        hidden_states.broadcast_mul(&self.layer_scalar)
    }
}

pub struct EmbeddingGemma2Model {
    embed_tokens: Embedding,
    embed_scale: f64,
    ple: EmbeddingGemma2PLE,
    layers: Vec<EmbeddingGemma2Layer>,
    norm: EmbeddingGemma2RMSNorm,
    embedding_projection: Linear,

    inv_freqs: HashMap<EmbeddingGemma2LayerType, Tensor>,
    sliding_window: usize,

    num_attention_heads: usize,
    pad_token_id: u32,
    pool: Pool,

    dtype: DType,
    device: Device,

    span: tracing::Span,
}

impl EmbeddingGemma2Model {
    pub fn load(
        vb: VarBuilder,
        config: &EmbeddingGemma2Config,
        model_type: ModelType,
    ) -> Result<Self> {
        let pool = match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for EmbeddingGemma2")
            }
            ModelType::Embedding(pool) => pool,
        };

        if matches!(pool, Pool::Cls | Pool::Splade) {
            candle::bail!("Only Mean and Last-Token pooling are supported for EmbeddingGemma2, no CLS or SPLADE");
        }

        let config = &config.text_config;
        let vb = vb.pp("language_model");

        if config.layer_types.len() != config.num_hidden_layers {
            candle::bail!(
                "`layer_types` has {} entries but `num_hidden_layers` is {}",
                config.layer_types.len(),
                config.num_hidden_layers
            );
        }

        let embed_tokens = Embedding::new(
            vb.pp("embed_tokens")
                .get((config.vocab_size, config.hidden_size), "weight")?,
            config.hidden_size,
        );
        let embed_scale = (config.hidden_size as f64).sqrt();

        let ple = EmbeddingGemma2PLE::load(vb.pp("ple"), config)?;

        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| {
                EmbeddingGemma2Layer::load(vb.pp(format!("layers.{layer_idx}")), config, layer_idx)
            })
            .collect::<Result<Vec<EmbeddingGemma2Layer>>>()?;

        let norm =
            EmbeddingGemma2RMSNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let embedding_projection_weight = vb
            .pp("embedding_projection")
            .get((config.embedding_dim, config.hidden_size), "weight")?;
        let embedding_projection = Linear::new(embedding_projection_weight, None, None);

        // NOTE: The full and sliding attention layers come with both a different RoPE theta and a
        // different head dimension, so the inverse frequencies are computed per layer type
        let mut inv_freqs = HashMap::new();
        for (layer_idx, layer_type) in config.layer_types.iter().enumerate() {
            if inv_freqs.contains_key(layer_type) {
                continue;
            }
            let rope_parameters = config.rope_parameters.get(layer_type).ok_or_else(|| {
                candle::Error::Msg(format!(
                    "`rope_parameters` is missing an entry for `{layer_type:?}`"
                ))
            })?;
            let head_dim = config.layer_config(layer_idx).head_dim;
            inv_freqs.insert(
                *layer_type,
                get_inv_freqs(head_dim, rope_parameters.rope_theta, vb.device(), None)?,
            );
        }

        Ok(Self {
            embed_tokens,
            embed_scale,
            ple,
            layers,
            norm,
            embedding_projection,
            inv_freqs,
            sliding_window: config.sliding_window,
            pool,
            pad_token_id: config.pad_token_id,
            num_attention_heads: config.num_attention_heads,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    /// Computes the cos and sin for the given position IDs on the fly rather than caching those up
    /// to `max_position_embeddings`, as that's 262144 positions, way beyond the 8192 context length
    fn get_cos_sin(
        &self,
        layer_type: EmbeddingGemma2LayerType,
        position_ids: &Tensor,
        batch_size: usize,
        max_length: usize,
    ) -> Result<(Tensor, Tensor)> {
        let inv_freqs = &self.inv_freqs[&layer_type];
        let positions = position_ids
            .flatten_all()?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let freqs = positions.matmul(inv_freqs)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        let rotary_dim = freqs.dim(1)?;

        let cos = freqs
            .cos()?
            .to_dtype(self.dtype)?
            .reshape((batch_size, 1, max_length, rotary_dim))?;
        let sin = freqs
            .sin()?
            .to_dtype(self.dtype)?
            .reshape((batch_size, 1, max_length, rotary_dim))?;
        Ok((cos, sin))
    }

    /// Bidirectional attention bias that masks the padding and, if provided, the tokens outside the
    /// sliding window. Note that `sliding_window` is a radius, so a token attends to every token
    /// within `sliding_window` positions in either direction (unlike Gemma3, where it's a width)
    fn get_attention_bias(
        &self,
        input_lengths: &[usize],
        max_length: usize,
        sliding_window: Option<usize>,
    ) -> Result<Option<Tensor>> {
        let has_padding = input_lengths.iter().any(|&length| length < max_length);
        let exceeds_window = sliding_window.is_some_and(|window| max_length > window + 1);
        if !has_padding && !exceeds_window {
            return Ok(None);
        }

        let batch_size = input_lengths.len();
        // NOTE: `f32::MIN` rather than `f32::NEG_INFINITY` so that fully masked rows (i.e. padding
        // tokens outside the sliding window of any non-padding token) don't produce NaNs
        let attention_bias: Vec<f32> = input_lengths
            .iter()
            .flat_map(|&length| {
                (0..max_length).flat_map(move |i| {
                    (0..max_length).map(move |j| {
                        let allowed = j < length
                            && sliding_window.is_none_or(|window| i.abs_diff(j) <= window);
                        if allowed {
                            0.0
                        } else {
                            f32::MIN
                        }
                    })
                })
            })
            .collect();

        let attention_bias = Tensor::from_vec(
            attention_bias,
            (batch_size, 1, max_length, max_length),
            &self.device,
        )?
        .to_dtype(self.dtype)?
        .broadcast_as((batch_size, self.num_attention_heads, max_length, max_length))?
        .contiguous()?;

        Ok(Some(attention_bias))
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let _enter = self.span.enter();

        let batch_size = batch.len();
        let max_length = batch.max_length as usize;

        let shape = (batch_size, max_length);

        let (input_ids, position_ids, input_lengths) = if batch_size > 1 {
            let elems = batch_size * max_length;

            let mut input_ids = Vec::with_capacity(elems);
            let mut position_ids = Vec::with_capacity(elems);
            let mut input_lengths = Vec::with_capacity(batch_size);

            for i in 0..batch_size {
                let start = batch.cumulative_seq_lengths[i] as usize;
                let end = batch.cumulative_seq_lengths[i + 1] as usize;
                let seq_length = end - start;
                input_lengths.push(seq_length);

                for j in start..end {
                    input_ids.push(batch.input_ids[j]);
                    position_ids.push(batch.position_ids[j]);
                }

                for _ in 0..(max_length - seq_length) {
                    input_ids.push(self.pad_token_id);
                    position_ids.push(0);
                }
            }

            let input_ids = Tensor::from_vec(input_ids, shape, &self.device)?;
            let position_ids = Tensor::from_vec(position_ids, shape, &self.device)?;

            (input_ids, position_ids, input_lengths)
        } else {
            let input_ids = Tensor::from_vec(
                batch.input_ids.clone(),
                (1, batch.input_ids.len()),
                &self.device,
            )?;
            let position_ids = Tensor::from_vec(
                batch.position_ids.clone(),
                (1, batch.position_ids.len()),
                &self.device,
            )?;
            let input_lengths = vec![batch.input_ids.len()];

            (input_ids, position_ids, input_lengths)
        };

        let full_attention_bias = self.get_attention_bias(&input_lengths, max_length, None)?;
        let sliding_attention_bias =
            self.get_attention_bias(&input_lengths, max_length, Some(self.sliding_window))?;

        let mut rotary = HashMap::new();
        for layer_type in self.inv_freqs.keys() {
            rotary.insert(
                *layer_type,
                self.get_cos_sin(*layer_type, &position_ids, batch_size, max_length)?,
            );
        }

        let inputs_embeds = (self.embed_tokens.forward(&input_ids)? * self.embed_scale)?;
        let per_layer_inputs = self.ple.forward(&inputs_embeds)?;

        let mut hidden_states = inputs_embeds;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let per_layer_input = per_layer_inputs.i((.., .., layer_idx))?;
            let attention_bias = match layer.layer_type {
                EmbeddingGemma2LayerType::FullAttention => full_attention_bias.as_ref(),
                EmbeddingGemma2LayerType::SlidingAttention => sliding_attention_bias.as_ref(),
            };
            let (cos, sin) = &rotary[&layer.layer_type];
            hidden_states =
                layer.forward(&hidden_states, &per_layer_input, attention_bias, cos, sin)?;
        }

        let outputs = self.norm.forward(&hidden_states)?;
        // NOTE: Projecting per token is equivalent to projecting after the mean pooling, and it
        // also makes the raw embeddings match the `token_embeddings` from Sentence Transformers
        let outputs = self.embedding_projection.forward(&outputs)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            match self.pool {
                Pool::Cls | Pool::Splade => {
                    unreachable!("Only Mean and Last-Token pooling are supported for EmbeddingGemma2, no CLS or SPLADE");
                }
                Pool::Mean => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .iter()
                            .map(|&i| {
                                let i = i as usize;
                                let length = input_lengths[i];
                                let embeddings = outputs.i((i, ..length))?;
                                embeddings.sum_keepdim(0)? / (length as f64)
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        let length = input_lengths[0];
                        let embeddings = outputs.i((0, ..length))?;
                        Some((embeddings.sum_keepdim(0)? / (length as f64))?)
                    }
                }
                Pool::LastToken => {
                    if batch_size > 1 {
                        let results: Result<Vec<Tensor>> = batch
                            .pooled_indices
                            .iter()
                            .map(|&i| {
                                let i = i as usize;
                                outputs.i((i..i + 1, input_lengths[i] - 1))
                            })
                            .collect();

                        Some(Tensor::cat(&results?, 0)?)
                    } else {
                        Some(outputs.i((.., max_length - 1))?)
                    }
                }
            }
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                let mut final_embeddings = Vec::new();
                for &i in &batch.raw_indices {
                    let i = i as usize;
                    let length = input_lengths[i];
                    final_embeddings.push(outputs.i((i, ..length))?);
                }
                Some(Tensor::cat(&final_embeddings, 0)?)
            } else {
                // Single batch or all raw requests
                if batch_size == 1 {
                    let length = input_lengths[0];
                    Some(outputs.i((0, ..length))?)
                } else {
                    // Multiple sequences, all raw
                    let mut all_embeddings = Vec::new();
                    for (i, &length) in input_lengths.iter().enumerate().take(batch_size) {
                        all_embeddings.push(outputs.i((i, ..length))?);
                    }
                    Some(Tensor::cat(&all_embeddings, 0)?)
                }
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

impl Model for EmbeddingGemma2Model {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
