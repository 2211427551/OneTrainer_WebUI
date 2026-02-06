import json
import logging
from typing import Any, Callable

import gradio as gr

from modules.webui.training_manager import TrainingManager

logger = logging.getLogger(__name__)

# --- CSS & Theme ---
CUSTOM_CSS = """
/* Light Theme Variables */
:root {
    --ot-bg: #ffffff;
    --ot-card-bg: #f9fafb;
    --ot-border: #e5e7eb;
    --ot-text: #1f2937;
    --ot-text-sub: #6b7280;
    --ot-accent: #3b82f6;
    --ot-accent-hover: #2563eb;
    --ot-success: #22c55e;
    --ot-warning: #eab308;
    --ot-error: #ef4444;
}

/* General Layout */
body, .gradio-container {
    background-color: var(--ot-bg) !important;
    color: var(--ot-text) !important;
}

/* Card Style */
.ot-card {
    background-color: var(--ot-card-bg) !important;
    border: 1px solid var(--ot-border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    margin-bottom: 12px;
}

/* Typography */
.ot-mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
.ot-header-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--ot-text);
}
.ot-label {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--ot-text-sub);
    margin-bottom: 4px;
}
.ot-value {
    font-size: 1.125rem;
    font-weight: 500;
    color: var(--ot-text);
}

/* Status Pills */
.ot-pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #f3f4f6;
    color: #374151;
    border: 1px solid #e5e7eb;
}
.ot-pill.running { background-color: #dcfce7; color: #166534; border-color: #bbf7d0; }
.ot-pill.stopped { background-color: #fee2e2; color: #991b1b; border-color: #fecaca; }
.ot-pill.idle { background-color: #f3f4f6; color: #6b7280; border-color: #e5e7eb; }

/* Progress Bar */
.ot-progress-container {
    width: 100%;
    height: 8px;
    background-color: #e5e7eb;
    border-radius: 9999px;
    overflow: hidden;
    margin-top: 8px;
    margin-bottom: 4px;
}
.ot-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--ot-accent), var(--ot-success));
    transition: width 0.5s ease;
}

/* Dashboard Grid */
.ot-dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
}
.ot-stat-card {
    background-color: var(--ot-card-bg);
    border: 1px solid var(--ot-border);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
}

/* Log Box */
.ot-log-box textarea {
    background-color: #f3f4f6 !important;
    color: #374151 !important; 
    font-family: ui-monospace, monospace !important;
    border: 1px solid var(--ot-border) !important;
    font-size: 0.85rem !important;
}

/* Form Styling */
.ot-group-label {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--ot-accent);
    border-bottom: 1px solid var(--ot-border);
    padding-bottom: 4px;
}
"""

OT_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size="sm",
    radius_size="md",
).set(
    body_background_fill="#ffffff",
    block_background_fill="#f9fafb",
    block_border_color="#e5e7eb",
    block_title_text_color="#1f2937",
    block_label_text_color="#6b7280",
    input_background_fill="#ffffff",
    input_border_color="#d1d5db",
    input_border_color_focus="#3b82f6",
)

# --- Help Text Dictionary ---
HELP_TEXTS = {
    # General Optimizer
    "optimizer.adam_w_mode": "Whether to use weight decay correction for Adam optimizer.",
    "optimizer.alpha": "Smoothing parameter for RMSprop and others.",
    "optimizer.amsgrad": "Whether to use the AMSGrad variant for Adam.",
    "optimizer.beta1": "Optimizer momentum term.",
    "optimizer.beta2": "Coefficients for computing running averages of gradient.",
    "optimizer.beta3": "Coefficient for computing the Prodigy stepsize.",
    "optimizer.bias_correction": "Whether to use bias correction in optimization algorithms like Adam.",
    "optimizer.block_wise": "Whether to perform block-wise model update.",
    "optimizer.capturable": "Whether some property of the optimizer can be captured.",
    "optimizer.centered": "Whether to center the gradient before scaling. Great for stabilizing the training process.",
    "optimizer.clip_threshold": "Clipping value for gradients.",
    "optimizer.dampening": "Dampening for optimizer momentum.",
    "optimizer.decay_rate": "Rate of decay for moment estimation.",
    "optimizer.decouple": "Use AdamW style decoupled weight decay.",
    "optimizer.differentiable": "Whether the optimization function is differentiable.",
    "optimizer.eps": "A small value to prevent division by zero.",
    "optimizer.eps2": "A small value to prevent division by zero.",
    "optimizer.foreach": "Whether to use a foreach implementation if available. Usually faster.",
    "optimizer.fsdp_in_use": "Flag for using sharded parameters.",
    "optimizer.fused": "Whether to use a fused implementation if available. Usually faster and requires less memory.",
    "optimizer.fused_back_pass": "Whether to fuse the back propagation pass with the optimizer step. Reduces VRAM usage, incompatible with gradient accumulation.",
    "optimizer.initial_accumulator_value": "Sets the starting value for moment estimates/Adagrad to ensure numerical stability.",
    "optimizer.is_paged": "Whether the optimizer's internal state should be paged to CPU.",
    "optimizer.log_every": "Intervals at which logging should occur.",
    "optimizer.lr_decay": "Rate at which learning rate decreases.",
    "optimizer.max_unorm": "Maximum value for gradient clipping by norms.",
    "optimizer.maximize": "Whether to maximize the optimization function.",
    "optimizer.momentum": "Factor to accelerate SGD in relevant direction.",
    "optimizer.nesterov": "Whether to enable Nesterov momentum.",
    "optimizer.no_prox": "Whether to use proximity updates or not.",
    "optimizer.optim_bits": "Number of bits used for optimization.",
    "optimizer.percentile_clipping": "Gradient clipping based on percentile values.",
    "optimizer.relative_step": "Whether to use a relative step size.",
    "optimizer.scale_parameter": "Whether to scale the parameter or not.",
    "optimizer.stochastic_rounding": "Stochastic rounding for weight updates. Improves quality when using bfloat16 weights.",
    "optimizer.use_triton": "Whether Triton optimization should be used.",
    "optimizer.warmup_init": "Whether to warm-up the optimizer initialization.",
    "optimizer.weight_decay": "Regularization to prevent overfitting.",
    "optimizer.weight_lr_power": "During warmup, the weights in the average will be equal to lr raised to this power. Set to 0 for no weighting.",
    
    # Advanced / Specific
    "optimizer.decoupled_decay": "If set as True, then the optimizer uses decoupled weight decay as in AdamW.",
    "optimizer.fixed_decay": "Applies fixed weight decay when True; scales decay with learning rate when False.",
    "optimizer.rectify": "Perform the rectified update similar to RAdam.",
    "optimizer.degenerated_to_sgd": "Performs SGD update when gradient variance is high.",
    "optimizer.d0": "Initial D estimate for D-adaptation.",
    "optimizer.d_coef": "Coefficient in the expression for the estimate of d.",
    "optimizer.growth_rate": "Limit for D estimate growth rate.",
    "optimizer.safeguard_warmup": "Avoid issues during warm-up stage.",
    "optimizer.slice_p": "Reduce memory usage by calculating LR adaptation statistics on only every pth entry of each tensor.",
    "optimizer.prodigy_steps": "Turn off Prodigy after N steps.",
    "optimizer.d_limiter": "Prevent over-estimated LRs when gradients and EMA are still stabilizing.",
    
    # Muon & Others
    "optimizer.MuonWithAuxAdam": "Whether to use the standard way of Muon. Non-hidden layers fallback to ADAMW.",
    "optimizer.muon_hidden_layers": "Comma-separated list of hidden layers to train using Muon.",
    "optimizer.muon_adam_lr": "Learning rate for the auxiliary AdamW optimizer.",
    "optimizer.rms_rescaling": "Integrates a more accurate method to match the Adam LR (slower but more accurate).",
    "optimizer.normuon_variant": "Enables the NorMuon optimizer variant (Muon orthogonalization + per-neuron adaptive learning rates).",
    "optimizer.low_rank_ortho": "Accelerate Muon by orthogonalizing only in a low-dimensional subspace.",
    "optimizer.use_AdEMAMix": "Adds a second, slow-moving EMA combined with the primary momentum.",
    "optimizer.kappa_p": "Controls the Lp-norm geometry for the Lion update (1.0 = Standard/Sign, 2.0 = Spherical).",
    "optimizer.ns_steps": "Controls the number of iterations for update orthogonalization. Higher values improve quality but are slower.",
    "optimizer.cautious_wd": "Applies weight decay only to parameter coordinates whose signs align with the optimizer update direction.",
    "optimizer.min_8bit_size": "Minimum tensor size for 8-bit quantization.",
    "optimizer.quant_block_size": "Size of a block of normalized 8-bit quantization data.",
    "optimizer.nnmf_factor": "Enables a memory-efficient mode by applying fast low-rank factorization to the optimizer states.",
    "optimizer.compile": "Enables PyTorch compilation for the optimizer internal step logic.",
    "optimizer.k": "Parameter for vector projections (used in AIDA).",
    "optimizer.xi": "Parameter for vector projections (used in AIDA).",
    "optimizer.adanorm": "Whether to use the AdaNorm variant.",
    "optimizer.orthogonal_gradient": "Reduces overfitting by removing the gradient component parallel to the weight.",
    "optimizer.use_atan2": "A robust replacement for eps, incorporating gradient clipping.",
    "optimizer.allora": "Scaling method for Automagic SinkGD optimizer.",
    "optimizer.auto_kappa_p": "Automatically determines optimal P-value for Lion (2.0 for 4D, 1.0 for 2D tensors).",
    
    # PEFT / General Training
    "lokr_factor": "Decomposition factor for LoKr. Controls the shape of Kronecker factors. -1 for auto.",
    "lora_rank": "Dimension of the low-rank matrices. Higher values mean more parameters.",
    "lora_alpha": "Scaling factor for LoRA updates. Usually set to equal rank or rank/2.",
    "train_batch_size": "Number of samples per training step.",
    "gradient_accumulation_steps": "Number of steps to accumulate gradients before updating weights.",
    "learning_rate": "The step size at each iteration while moving toward a minimum of a loss function.",
}


class ConfigBuilder:
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.fields = schema.get("fields", []) if isinstance(schema, dict) else []
        self.path_map: dict[str, dict] = {f["path"]: f for f in self.fields if "path" in f}
        self.components: list[Any] = [] # Ordered list of components for saving
        self.component_paths: list[str] = [] # Corresponding paths
        self.path_to_comp: dict[str, Any] = {} # Map path to component
        self.model_type_comp = None
        self.base_model_comp = None
        self.peft_type_comp = None
        self.optimizer_comp = None

    def _get_category(self, path: str) -> str:
        if path.startswith("model.") or path.startswith("lora") or path.startswith("peft") or path.startswith("lokr") or path.startswith("oft"):
            return "Model/PEFT"
        if path.startswith("optimizer"):
            return "Optimizer"
        if path.startswith("concepts") or path == "concept_file_name":
            return "Concepts"
        if path.startswith("samples") or path.startswith("sample"):
            return "Samples"
        if path.startswith("save") or path.startswith("backup"):
            return "Saving"
        if path.startswith("secrets"):
            return "Cloud/Secrets"
        if "." not in path:
            return "Training"
        return "Advanced"

    def _format_label(self, path: str) -> str:
        if "." in path:
            parts = path.split(".")
            name = parts[-1]
        else:
            name = path
        return name.replace("_", " ").title()

    def build_ui(self):
        # Group fields
        groups: dict[str, list[dict]] = {
            "Training": [], "Model/PEFT": [], "Optimizer": [], "Concepts": [], 
            "Samples": [], "Saving": [], "Cloud/Secrets": [], "Advanced": []
        }
        
        sorted_fields = sorted(self.fields, key=lambda x: x.get("path", ""))
        
        for f in sorted_fields:
            path = f.get("path")
            if not path: continue
            cat = self._get_category(path)
            groups[cat].append(f)

        # Create Tabs
        category_order = ["Training", "Model/PEFT", "Optimizer", "Concepts", "Samples", "Saving", "Cloud/Secrets", "Advanced"]
        
        for cat in category_order:
            fields = groups.get(cat)
            if not fields: continue
            
            with gr.Tab(cat):
                # 3-Column Grid Layout
                cols = [[], [], []]
                for i, f in enumerate(fields):
                    cols[i % 3].append(f)
                
                with gr.Row():
                    with gr.Column():
                        for f in cols[0]:
                            self._create_input(f)
                    with gr.Column():
                        for f in cols[1]:
                            self._create_input(f)
                    with gr.Column():
                        for f in cols[2]:
                            self._create_input(f)

        # --- Dynamic Logic Bindings ---
        
        # 1. Model Type Auto-Detection
        if self.base_model_comp and self.model_type_comp:
            def detect_model_type(base_model_path: str):
                if not base_model_path: return gr.update()
                lower = base_model_path.lower()
                if "sdxl" in lower: return "SDXL"
                if "v1-5" in lower or "1.5" in lower: return "SD_1_5"
                if "v2" in lower or "2.1" in lower: return "SD_2_1"
                if "flux" in lower: return "FLUX"
                return gr.update()

            self.base_model_comp.change(fn=detect_model_type, inputs=[self.base_model_comp], outputs=[self.model_type_comp])

        # 2. PEFT Visibility Logic
        if self.peft_type_comp:
            controlled_paths = [
                "lokr_factor", 
                "oft_block_size", "oft_coft", "coft_eps", "oft_block_share",
                "lora_rank", "lora_alpha", "lora_decompose", "lora_decompose_norm_epsilon", "lora_decompose_output_axis", "lora_weight_dtype", "lora_model_name"
            ]
            controlled_comps = [self.path_to_comp[p] for p in controlled_paths if p in self.path_to_comp]
            
            def peft_logic(peft_type):
                # Returns list of updates matching controlled_comps order
                vis = {p: False for p in controlled_paths}
                
                if peft_type == "LOKR":
                    vis["lokr_factor"] = True
                    vis["lora_rank"] = True
                    vis["lora_alpha"] = True
                    vis["lora_weight_dtype"] = True
                    vis["lora_model_name"] = True
                elif peft_type == "OFT":
                    vis["oft_block_size"] = True
                    vis["oft_coft"] = True
                    vis["coft_eps"] = True
                    vis["oft_block_share"] = True
                    vis["lora_model_name"] = True 
                elif peft_type == "LORA" or peft_type == "LOHA" or peft_type == "LOCON" or peft_type == "LORA_FA": 
                    for p in ["lora_rank", "lora_alpha", "lora_decompose", "lora_decompose_norm_epsilon", "lora_decompose_output_axis", "lora_weight_dtype", "lora_model_name"]:
                        vis[p] = True
                
                return [gr.update(visible=vis.get(p, False)) for p in controlled_paths if p in self.path_to_comp]

            self.peft_type_comp.change(
                fn=peft_logic,
                inputs=[self.peft_type_comp],
                outputs=controlled_comps
            )

        # 3. Optimizer Visibility Logic
        if self.optimizer_comp:
            # Define optimizer specific params
            # Map based on OneTrainer optimizer implementations
            opt_map = {
                "PRODIGY": ["optimizer.prodigy_steps", "optimizer.d0", "optimizer.d_coef", "optimizer.growth_rate", "optimizer.use_bias_correction", "optimizer.safeguard_warmup", "optimizer.d_limiter", "optimizer.weight_decay", "optimizer.weight_decay_by_lr"],
                "DADAPTATION": ["optimizer.d0", "optimizer.d_coef", "optimizer.growth_rate", "optimizer.weight_decay"],
                "ADAMW": ["optimizer.beta1", "optimizer.beta2", "optimizer.eps", "optimizer.weight_decay", "optimizer.weight_decay_by_lr", "optimizer.amsgrad", "optimizer.bias_correction"],
                "ADAMW_8BIT": ["optimizer.beta1", "optimizer.beta2", "optimizer.eps", "optimizer.weight_decay", "optimizer.weight_decay_by_lr", "optimizer.bias_correction"],
                "LION": ["optimizer.beta1", "optimizer.beta2", "optimizer.weight_decay", "optimizer.weight_decay_by_lr", "optimizer.use_cautious", "optimizer.kappa_p", "optimizer.auto_kappa_p"],
                "LION_8BIT": ["optimizer.beta1", "optimizer.beta2", "optimizer.weight_decay", "optimizer.weight_decay_by_lr", "optimizer.use_cautious", "optimizer.kappa_p", "optimizer.auto_kappa_p"],
                "ADAFACTOR": ["optimizer.relative_step", "optimizer.scale_parameter", "optimizer.warmup_init", "optimizer.weight_decay"],
                "MUON": ["optimizer.muon_hidden_layers", "optimizer.MuonWithAuxAdam", "optimizer.muon_adam_lr", "optimizer.muon_te1_adam_lr", "optimizer.muon_te2_adam_lr", "optimizer.normuon_variant", "optimizer.ns_steps"],
                "MUON_ADV": ["optimizer.muon_hidden_layers", "optimizer.MuonWithAuxAdam", "optimizer.muon_adam_lr", "optimizer.normuon_variant", "optimizer.ns_steps", "optimizer.cautious_wd", "optimizer.accelerated_ns"],
                "ADAMUON_ADV": ["optimizer.muon_hidden_layers", "optimizer.MuonWithAuxAdam", "optimizer.muon_adam_lr"],
                "AUTOMAGIC_SINKGD": ["optimizer.allora", "optimizer.eta", "optimizer.orthograd", "optimizer.sinkgd_iters"],
                "SCHEDULEFREE_ADAMW": ["optimizer.use_schedulefree", "optimizer.schedulefree_c", "optimizer.k_warmup_steps", "optimizer.beta1", "optimizer.beta2", "optimizer.weight_decay"],
                "SCHEDULEFREE_SGD": ["optimizer.use_schedulefree", "optimizer.schedulefree_c", "optimizer.k_warmup_steps", "optimizer.momentum", "optimizer.weight_decay"],
                "ADEMAMIX": ["optimizer.use_AdEMAMix", "optimizer.beta3_ema", "optimizer.alpha_grad", "optimizer.beta1_warmup", "optimizer.min_beta1", "optimizer.Simplified_AdEMAMix"]
            }
            
            # Collect all potentially hidden optimizer fields
            all_opt_paths = set(p for sublist in opt_map.values() for p in sublist)
            # Add other specific fields that should be hidden by default
            extra_hidden = [
                "optimizer.momentum", "optimizer.nesterov", "optimizer.rho", "optimizer.centered", 
                "optimizer.rms_rescaling", "optimizer.low_rank_ortho", "optimizer.ortho_rank",
                "optimizer.approx_mars", "optimizer.use_kahan", "optimizer.use_grams", "optimizer.grams_moment",
                "optimizer.kourkoutas_beta", "optimizer.cautious", "optimizer.cautious_mask",
                "optimizer.use_stableadamw", "optimizer.use_orthograd", "optimizer.orthogonal_gradient",
                "optimizer.use_atan2", "optimizer.use_adopt", "optimizer.nnmf_factor", "optimizer.factored",
                "optimizer.adanorm", "optimizer.ams_bound", "optimizer.adam_debias", "optimizer.rectify",
                "optimizer.degenerated_to_sgd"
            ]
            all_opt_paths.update(extra_hidden)
            
            opt_controlled_comps = [self.path_to_comp[p] for p in all_opt_paths if p in self.path_to_comp]
            
            # Helper to map component to its path
            comp_to_path = {self.path_to_comp[p]: p for p in all_opt_paths if p in self.path_to_comp}
            
            def opt_logic(opt_name):
                # Default to showing nothing from the specific list if unknown
                target_paths = opt_map.get(opt_name, [])
                
                updates = []
                for comp in opt_controlled_comps:
                    path = comp_to_path.get(comp)
                    is_visible = path in target_paths
                    updates.append(gr.update(visible=is_visible))
                return updates

            self.optimizer_comp.change(
                fn=opt_logic,
                inputs=[self.optimizer_comp],
                outputs=opt_controlled_comps
            )

    def _create_input(self, field: dict):
        path = field["path"]
        kind = field.get("kind", "str")
        default = field.get("default")
        nullable = field.get("nullable", False)
        enum_vals = field.get("enum")
        
        label = self._format_label(path)
        info_text = HELP_TEXTS.get(path)
        
        comp = None
        if kind == "bool":
            comp = gr.Checkbox(label=label, value=default if default is not None else False, info=info_text)
        elif kind == "int":
            comp = gr.Number(label=label, value=default, precision=0, info=info_text)
        elif kind == "float":
            comp = gr.Number(label=label, value=default, info=info_text)
        elif kind == "enum" and enum_vals:
            comp = gr.Dropdown(label=label, choices=enum_vals, value=default, info=info_text)
        elif kind in ("list", "dict", "json"):
            # Use Code editor for complex types
            val_str = json.dumps(default, indent=2) if default is not None else ""
            comp = gr.Code(label=label, value=val_str, language="json", lines=3) # Code component doesn't support info=
        else:
            comp = gr.Textbox(label=label, value=str(default) if default is not None else "", info=info_text)

        # Store special components for logic
        if path == "model.model_type":
            self.model_type_comp = comp
        elif path == "model.name_or_path":
            self.base_model_comp = comp
        elif path == "peft_type":
            self.peft_type_comp = comp
        elif path == "optimizer.optimizer":
            self.optimizer_comp = comp

        self.components.append(comp)
        self.component_paths.append(path)
        self.path_to_comp[path] = comp
        return comp

    def get_config_from_ui(self, *values) -> dict[str, Any]:
        """Reconstruct config dict from flat list of values."""
        cfg = {}
        for path, val in zip(self.component_paths, values):
            # Parse value based on schema kind
            field = self.path_map.get(path)
            kind = field.get("kind", "str")
            
            parsed = val
            if kind in ("list", "dict", "json"):
                try:
                    parsed = json.loads(val) if val.strip() else None
                except:
                    parsed = None # Or keep as string? Better to fail safely
            elif kind == "int":
                parsed = int(val) if val is not None else None
            elif kind == "float":
                parsed = float(val) if val is not None else None
            
            # Set deep value
            self._set_deep(cfg, path, parsed)
        return cfg

    def get_ui_values_from_config(self, config: dict[str, Any]) -> list[Any]:
        """Flatten config dict to list of values matching components."""
        values = []
        for path in self.component_paths:
            val = self._get_deep(config, path)
            field = self.path_map.get(path)
            kind = field.get("kind", "str")
            
            if kind in ("list", "dict", "json"):
                values.append(json.dumps(val, indent=2, ensure_ascii=False) if val is not None else "")
            else:
                values.append(val)
        return values

    def _set_deep(self, d: dict, path: str, value: Any):
        parts = path.split(".")
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    def _get_deep(self, d: dict, path: str) -> Any:
        parts = path.split(".")
        cur = d
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur


def create_gradio_app(manager: TrainingManager, project_root: str) -> gr.Blocks:
    
    # --- Data Helpers ---
    def _get_dashboard_html(state: dict, stats: dict) -> str:
        run_id = state.get("run_id") or "N/A"
        status = state.get("status", "idle")
        
        # Progress
        progress = state.get("progress", {})
        gs = progress.get("global_step", 0) or 0
        max_step = progress.get("max_step") or 100
        pct = 0
        if max_step > 0:
            pct = min(100, max(0, gs / max_step * 100))
        
        # Speed
        speed = state.get("speed", {})
        sps = speed.get("steps_per_sec")
        sps_text = f"{sps:.2f} it/s" if sps else "-"
        
        # Stats
        cpu = stats.get("cpu_percent", 0) or 0
        mem = (stats.get("mem") or {}).get("used_percent", 0) or 0
        gpu_info = stats.get("gpus", [])
        gpu_html = ""
        for g in gpu_info:
            gpu_html += f"""
            <div class="ot-stat-card">
                <div class="ot-label">{g.get('name', 'GPU')}</div>
                <div class="ot-value">{g.get('util_gpu', 0)}%</div>
                <div class="ot-sub" style="font-size: 0.75rem; color: #6b7280;">{g.get('temp_c', 0)}°C | {g.get('mem_used_mb', 0)}MB</div>
            </div>
            """
        
        status_class = "running" if status == "starting" or state.get("running") else ("stopped" if status == "failed" or status == "finished" else "idle")

        return f"""
        <div class="ot-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <span class="ot-header-title">{run_id}</span>
                    <span class="ot-pill {status_class}" style="margin-left: 8px;">{status.upper()}</span>
                </div>
                <div class="ot-mono" style="color: var(--ot-accent); font-weight: 600;">{sps_text}</div>
            </div>
            
            <div class="ot-progress-container">
                <div class="ot-progress-bar" style="width: {pct}%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--ot-text-sub); margin-bottom: 16px;">
                <span>Step {gs} / {max_step}</span>
                <span>{pct:.1f}%</span>
            </div>
            
            <div class="ot-dashboard-grid">
                <div class="ot-stat-card">
                    <div class="ot-label">CPU Load</div>
                    <div class="ot-value">{cpu}%</div>
                </div>
                <div class="ot-stat-card">
                    <div class="ot-label">RAM Usage</div>
                    <div class="ot-value">{mem}%</div>
                </div>
                {gpu_html}
            </div>
        </div>
        """

    def refresh_dashboard(run_id_val, tail_bytes):
        state = manager.get_state()
        stats = manager.get_system_stats()
        
        # Log
        choices = [r["run_id"] for r in manager.list_runs()]
        active_run_id = state.get("run_id")
        selected_run_id = run_id_val or active_run_id or (choices[0] if choices else None)
        
        log_content = ""
        if selected_run_id:
            log_content = manager.read_log_tail(selected_run_id, int(tail_bytes))
            
        html = _get_dashboard_html(state, stats)
        
        # Checkpoints
        ckpts = manager.list_checkpoints(project_root, limit=5)
        ckpt_text = "\n".join([f"• {c['path'].split('/')[-1]} ({c['size_mb']}MB)" for c in ckpts])
        
        return (
            html, 
            log_content, 
            ckpt_text,
            gr.Dropdown(choices=choices, value=selected_run_id)
        )

    # --- Schema Loading ---
    schema = manager.get_train_config_schema(project_root)
    if not schema or not schema.get("fields"):
        # Fallback schema if generation failed
        logger.error("Failed to load schema, using fallback.")
        schema = {"fields": []} # Should probably show an error UI
        
    builder = ConfigBuilder(schema)

    # --- Handlers ---
    def save_config_ui(*values):
        cfg = builder.get_config_from_ui(*values)
        manager.save_config(cfg)
        return "Config saved successfully."

    def load_config_ui():
        cfg = manager.load_config() or {}
        return builder.get_ui_values_from_config(cfg)

    def start_training():
        try:
            manager.start_training(project_root)
            return "Training started."
        except Exception as e:
            return f"Error: {e}"

    def stop_training():
        manager.stop_training()
        return "Stopping..."

    # --- UI Layout ---
    with gr.Blocks(title="OneTrainer WebUI", theme=OT_THEME, css=CUSTOM_CSS) as demo:
        
        with gr.Tab("Dashboard"):
            dashboard_html = gr.HTML(label="Status")
            
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row(variant="compact"):
                        run_select = gr.Dropdown(label="Select Run Log", choices=[], scale=2)
                        tail_slider = gr.Slider(label="Log Tail Size", minimum=1024, maximum=100000, value=20000, step=1000, scale=1)
                        btn_refresh = gr.Button("Refresh", variant="secondary", scale=0, min_width=80)
                    
                    log_view = gr.Code(label="Training Log", language=None, lines=20, elem_classes=["ot-log-box"])
                
                with gr.Column(scale=1):
                    gr.Markdown("### Actions", elem_classes=["ot-header-title"])
                    btn_start = gr.Button("Start Training", variant="primary")
                    btn_stop = gr.Button("Stop Training", variant="stop")
                    
                    gr.Markdown("### Recent Checkpoints", elem_classes=["ot-header-title"])
                    ckpt_list = gr.Textbox(label="", lines=10, interactive=False)
            
            # Auto-refresh
            timer = gr.Timer(2.0)
            timer.tick(refresh_dashboard, inputs=[run_select, tail_slider], outputs=[dashboard_html, log_view, ckpt_list, run_select])
            btn_refresh.click(refresh_dashboard, inputs=[run_select, tail_slider], outputs=[dashboard_html, log_view, ckpt_list, run_select])
            
            # Bind Actions
            btn_start.click(start_training, outputs=None)
            btn_stop.click(stop_training, outputs=None)

        with gr.Tab("Configuration"):
            with gr.Row():
                btn_load = gr.Button("Load Config from File", variant="secondary")
                btn_save = gr.Button("Save Config to File", variant="primary")
                status_msg = gr.Label(label="Status", value="", show_label=False)
            
            # Build the Form
            if not builder.fields:
                gr.Error("Could not load OneTrainer configuration schema. Please check console logs.")
            else:
                builder.build_ui()
            
            # Bind Load/Save
            btn_load.click(load_config_ui, outputs=builder.components)
            btn_save.click(save_config_ui, inputs=builder.components, outputs=status_msg)

        with gr.Tab("Dataset Tools"):
             btn_check = gr.Button("Check Dataset Integrity")
             check_out = gr.JSON(label="Report")
             btn_check.click(lambda: manager.dataset_check(project_root), outputs=check_out)

    return demo
