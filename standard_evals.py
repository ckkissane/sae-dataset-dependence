# %%
from utils import *
from sae_lens.training.activations_store import ActivationsStore
import wandb
from sae_lens.sae import SAE
from sae_lens.evals import run_evals, EvalConfig
torch.set_grad_enabled(False)
# %%
device = 'cuda:0'
model = HookedTransformer.from_pretrained(
    "qwen1.5-0.5b-chat", 
    device=device, 
    default_padding_side='left', 
)

model.tokenizer.padding_side = 'left'
model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# %%
# Load in base and chat SAEs

entity = "ckkissane"

project_to_arfiacts = {
    "qwen-500M-chat-lmsys-1m-anthropic": ["sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768:v11",],
    "qwen-500M-chat-pile-inst-format-with-assistant": ["sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768:v3"],
}

saes = {}

for project in project_to_arfiacts.keys():
    for artifact in project_to_arfiacts[project]:
        if project =="qwen-500M-chat-lmsys-1m-anthropic" and artifact == "sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768:v11":
            sae_name = "chat_lmsys_32768"
            estimated_norm_scaling_factor = 1.5326804001319736
        elif project == "qwen-500M-chat-pile-inst-format-with-assistant" and artifact == "sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768:v3":
            sae_name = "chat_pile_inst_format_with_assistant_32768"
            estimated_norm_scaling_factor = 1.573157268076237
        else:
            raise ValueError(f"Unknown project and artifact combination: {project}, {artifact}")
            
        artifact_path = f"{entity}/{project}/{artifact}"
        api = wandb.Api()
        artifact = api.artifact(artifact_path)
        artifact.download(root=f"./artifacts/{project}/{artifact}")
        sae = SAE.load_from_pretrained(f"./artifacts/{project}/{artifact}", device=device)
        
        # I think we need this due to SAE Lens bug
        sae.b_dec.data /= estimated_norm_scaling_factor 
        
        # fold decoder norms
        decoder_norms = sae.W_dec.data.norm(dim=-1, keepdim=True)
        sae.W_enc.data *= decoder_norms.T
        sae.b_enc.data *= decoder_norms.squeeze()
        sae.W_dec.data /= decoder_norms
        saes[sae_name] = sae

# %%
lmsys_sae = saes['chat_lmsys_32768']
pile_sae = saes['chat_pile_inst_format_with_assistant_32768']
# %%
# choose SAE based on dataset you want to evaluate
activations_store = ActivationsStore.from_sae(
    model=model,
    sae=lmsys_sae, # change this to pile_sae to evaluate on pile data
    streaming=True,
    store_batch_size_prompts=8,
    n_batches_in_buffer=8,
    device=device,
)

# %%
eval_config = EvalConfig(
    batch_size_prompts = 4,
    n_eval_reconstruction_batches = 5,
    compute_kl = False,
    compute_ce_loss = True,
    n_eval_sparsity_variance_batches = 5,
    compute_l2_norms = False,
    compute_sparsity_metrics = True,
    compute_variance_metrics = True, 
)
metrics = run_evals(
    lmsys_sae, # change this to pile_sae to evaluate the SAE trained on the pile
    activations_store,
    model,
    eval_config,
)
# %%
print(metrics)
# %%
