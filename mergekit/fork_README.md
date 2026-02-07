# Mergekit: Advanced MoE Upcycling & Continual Training Edition

This fork of `mergekit` extends the base Mixture-of-Experts (MoE) upcycling capabilities with state-of-the-art initialization and fusion techniques. It is specifically designed to transform dense monolingual models into robust, multilingual MoE architectures while mitigating **Expert Collapse** and **Catastrophic Forgetting**.

---

## 🚀 Key Features

### 1. Orthogonal Gate Initialization
Introduces `torch.nn.init.orthogonal` for MoE gate parameters.
- **The Problem**: Standard Gaussian (random) initialization often causes gate vectors to be highly correlated, leading to "Expert Collapse" where only a few experts are utilized.
- **The Solution**: Ensures the gate matrix starts with a condition number of 1, forcing experts to cover maximally distinct regions of the hidden state manifold.
- **Benefit**: Faster specialization during the initial multilingual fine-tuning phase.



### 2. MoE-CT (Continual Training) Fusion
A comprehensive weight fusion system to protect the base model's original reasoning capabilities during expansion.

* **Residual-Expert Fusion**: Blends base model FFN weights into new experts using the `base_alpha` parameter.
* **Weighted Gate Fusion**: Implements a `router_aux_scale` (default 1.1) to bias the router toward "Stability Anchors," ensuring the model defaults to reliable logic when uncertain.

### 3. Dynamic Alpha Scaling (U-Shaped Strategy)
A layer-wise scheduler that balances **Stability** (base knowledge) and **Plasticity** (new language learning) based on Transformer depth.
- **Lower/Upper Layers**: Higher alpha ($\alpha$) to allow adaptation of token embeddings and output styling.
- **Middle Layers**: Lower alpha ($\alpha \approx 0$) to protect the "Semantic Core" where deep reasoning and world knowledge are stored.



---

## 🛠 Supported Architectures

This fork provides optimized implementations for:
* **QwenMoE (Qwen 2 / 2.5)**: Supports shared expert anchoring.
* **Qwen3MoE**: Specialized for **Ultra-Sparse** configurations (up to 128 experts) with distributed base-model fusion.

---

## 📖 Usage

To utilize these features, update your MoE merge configuration with the following experimental fields:

```yaml
base_model: path/to/dense/base_model
gate_mode: orthogonal         # Maximizes expert diversity
moe_ct_mode: true             # Enables weight and gate fusion
base_alpha: 0.8               # Maximum plasticity for experts
alpha_strategy: u_shaped      # Preserves middle-layer reasoning
router_aux_scale: 1.1         # Bias toward original reasoning
```

## 🧪 Technical Implementation Details

### Weight Blending Engine

The core of the MoE-CT implementation is a high-precision weight blending engine. It uses linear interpolation (`torch.lerp`) to ensure numerical stability when operating in `bfloat16` or `float16` precision. This method prevents the "expert drift" typically seen when naive weight copying is used during upcycling.

The mathematical foundation for each fused parameter is:
$$W_{fused} = (1 - \alpha) \cdot W_{base} + \alpha \cdot W_{expert}$$



### The Alpha Scheduler

The `u_shaped` strategy is the primary mechanism for balancing linguistic adaptation with semantic stability. It acknowledges that the "knowledge" of a Transformer is not distributed uniformly.

- **Token-Level Plasticity:** Lower layers adapt to the specific token distributions of new languages.
- **Output-Level Plasticity:** Upper layers adapt to the syntax and style requirements.
- **Semantic Stability:** Middle layers act as a frozen anchor for logic and reasoning.

The dynamic alpha for each layer is calculated using a cosine-based decay:

```python
# layer_idx: current layer number
# total_layers: depth of the model
# depth: normalized depth (0.0 to 1.0)

depth = layer_idx / (total_layers - 1)
curve = 0.5 * (1 + math.cos(2 * math.pi * depth))
current_alpha = base_alpha * curve
```

### Weighted Gate Fusion

To prevent the router from defaulting to specialized experts for general reasoning tasks, we implement a Router Bias. This modifies the router's logits before the Top-K selection process.

By applying a router_aux_scale (typically between 1.05 and 1.2), we mathematically favor the experts that contain the highest concentration of "Anchor" (base model) knowledge. This acts as a confidence-based fallback mechanism.

## 🚦 Development & QA

### Verification Tests

We have included a comprehensive test suite to ensure mathematical integrity across the pipeline:

```bash
# Verify the weight fusion logic mathematically
uv run python -m unittest tests/test_moe_ct_fusion.py

# Verify the U-shaped alpha scheduler curve across 32 layers
uv run python -m unittest tests/test_alpha_scheduler.py

# Verify the integrated pipeline (Weights + Router Bias interaction)
uv run python -m unittest tests/test_moe_full_pipeline.py
```

### Installation

```bash
git clone [https://github.com/joenaess/mergekit.git](https://github.com/joenaess/mergekit.git)
cd mergekit
uv sync
```
