# %%
from utils import *
from transformer_lens import utils

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from sae_lens.sae import SAE
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
# Load SAEs

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
# Get true refusal direction
harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()
# %%
template = QWEN_CHAT_TEMPLATE
tokenize_instructions_fn = functools.partial(tokenize_instructions, tokenizer=model.tokenizer, template=template)
# %%
n_inst_train = 64
harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:n_inst_train])
harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:n_inst_train])
# %%
layer = 13
pos = -1

_, harmful_cache = model.run_with_cache(harmful_toks, names_filter=utils.get_act_name('resid_pre', layer), stop_at_layer=layer+1)
_, harmless_cache = model.run_with_cache(harmless_toks, names_filter=utils.get_act_name('resid_pre', layer), stop_at_layer=layer+1)

harmful_acts = harmful_cache['resid_pre', layer][:, pos, :]
harmless_acts = harmless_cache['resid_pre', layer][:, pos, :]

harmful_mean_act = harmful_acts.mean(dim=0)
harmless_mean_act = harmless_acts.mean(dim=0)

chat_refusal_dir = harmful_mean_act - harmless_mean_act

# %%
def get_avg_latent_diff(sae: SAE, debug : bool = True) -> torch.Tensor:
    harmless_sae_acts = sae.encode(harmless_acts)
    harmful_sae_acts = sae.encode(harmful_acts)
    
    avg_latent_diff = (harmful_sae_acts - harmless_sae_acts).mean(0)
    if debug:
        alternate_recons_chat = avg_latent_diff @ sae.W_dec
        recons_harmful_acts = sae(harmful_acts)
        recons_harmless_acts = sae(harmless_acts)

        recons_harmful_mean_act = recons_harmful_acts.mean(dim=0)
        recons_harmless_mean_act = recons_harmless_acts.mean(dim=0)
        
        recons_chat_refusal_dir = recons_harmful_mean_act - recons_harmless_mean_act
        assert torch.allclose(alternate_recons_chat, recons_chat_refusal_dir, atol=1e-3)
    return avg_latent_diff

for sae_name, sae in saes.items():
    print(sae_name)
    avg_latent_diff = get_avg_latent_diff(sae)
    print((avg_latent_diff != 0).sum())
    print("top positive", avg_latent_diff.topk(10))
    print("top negative", (-avg_latent_diff).topk(10))
    print()

# %%
torch.set_grad_enabled(True)

def fit_selected_rows(chat_refusal_dir, sae, latent_indices, num_iterations=1000, learning_rate=0.01):
    # Extract the selected rows from W_dec
    selected_rows = sae.W_dec[latent_indices].clone().detach()

    # Initialize coefficients
    coefficients = nn.Parameter(torch.randn(len(latent_indices), device=sae.device))

    # Define the optimizer
    optimizer = optim.Adam([coefficients], lr=learning_rate)

    # Training loop
    for i in range(num_iterations):
        # Compute the weighted sum
        weighted_sum = torch.sum(coefficients.unsqueeze(1) * selected_rows, dim=0)

        # Compute the loss (mean squared error)
        loss = torch.mean((weighted_sum - chat_refusal_dir).pow(2))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}/{num_iterations}, Loss: {loss.item():.6f}")

    final_weighted_sum = torch.sum(coefficients.unsqueeze(1) * selected_rows, dim=0)
    final_relative_squared_error = (final_weighted_sum - chat_refusal_dir).pow(2).sum() / chat_refusal_dir.pow(2).sum()
    final_loss = torch.mean(final_relative_squared_error)

    print(f"Final Loss: {final_loss.item():.6f}")
    return final_loss.item()

sae_names = ["chat_lmsys_32768", "chat_pile_inst_format_with_assistant_32768"]
records = []
for k in 2 ** torch.arange(0, 10):
    # for sae_name, sae in saes.items():
    for sae_name in sae_names:
        print(sae_name)
        sae = saes[sae_name]
        avg_latent_diff = get_avg_latent_diff(sae)
        latent_indices = avg_latent_diff.abs().topk(k).indices
        
        final_loss = fit_selected_rows(chat_refusal_dir, sae, latent_indices, num_iterations=1_000, learning_rate=0.01)
        
        records.append({
            "sae_name": sae_name,
            "k": k.item(),
            "final_loss": final_loss
        })

torch.set_grad_enabled(False)
# %%
records_df = pd.DataFrame(records)
records_df
# %%
fig = px.scatter(
    records_df,
    x='k',
    y='final_loss',
    color='sae_name',
    title="Sparse Linear regression: MSE loss vs. number of latents used"
)

fig.data[0].name = "LmSys"
fig.data[1].name = "Pile"
fig.update_layout(
    legend_title_text="SAE training dataset",
    xaxis_title="Number of latents used",
    yaxis_title="Relative MSE loss",
)

fig.show()

# %%
