# %%
import torch
import wandb
from sae_lens.sae import SAE
from utils import *
from transformer_lens import HookedTransformer
from sae_lens.training.activations_store import ActivationsStore
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner
torch.set_grad_enabled(False);

device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
# Load in SAEs
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
# Get SAEs
lmsys_sae = saes['chat_lmsys_32768']
pile_sae = saes["chat_pile_inst_format_with_assistant_32768"]
# %%
# Load in model
device = 'cuda:0'
model = HookedTransformer.from_pretrained(
    "qwen1.5-0.5b-chat", 
    device=device, 
    default_padding_side='left', 
    dtype=torch.bfloat16
)

model.tokenizer.padding_side = 'left'
model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# %%
# import activations store
activations_store = ActivationsStore.from_sae(
    model=model,
    sae=lmsys_sae, # change this if you don't want LmSys data used
    streaming=True,
    store_batch_size_prompts=8,
    n_batches_in_buffer=8,
    device=device,
)

# %%
torch.manual_seed(0)
def get_tokens(
    activations_store: ActivationsStore,
    n_prompts: int,
):
    all_tokens_list = []
    pbar = tqdm(range(n_prompts))
    for _ in pbar:
        batch_tokens = activations_store.get_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens

token_dataset = get_tokens(activations_store, 256)
# %%
token_dataset.shape
# %%
token_dataset.numel()
# %%
import gc

gc.collect()
torch.cuda.empty_cache()
# %%

# latent_indices = [25840, 16770, 11816] # lmsys SAE latents
latent_indices = [9542, 26531, 12276, 25271] # pile SAE latents

feature_vis_config_gpt = SaeVisConfig(
    hook_point=lmsys_sae.cfg.hook_name,
    features=latent_indices,
    minibatch_size_features=10,
    minibatch_size_tokens=8,  # this is number of prompts at a time.
    verbose=True,
    device="cuda",
    dtype="bfloat16",
)

data = SaeVisRunner(feature_vis_config_gpt).run(
    encoder=pile_sae,  # change this to use the other SAE
    model=model,
    tokens=token_dataset,
)

# %%
from sae_dashboard.data_writing_fns import save_feature_centric_vis

filename = f"pile_sae_latent_dashboards.html"
save_feature_centric_vis(sae_vis_data=data, filename=filename)
# %%
