# %%
from utils import *
from transformer_lens import utils

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
# now try with harmful and harmless instructions
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
def get_recons_refusal_dir_cos_sim(sae: SAE, chat_refusal_dir: torch.Tensor, debug: bool = False) -> float:
    recons_harmful_acts = sae(harmful_acts)
    recons_harmless_acts = sae(harmless_acts)

    recons_harmful_mean_act = recons_harmful_acts.mean(dim=0)
    recons_harmless_mean_act = recons_harmless_acts.mean(dim=0)
    
    recons_chat_refusal_dir = recons_harmful_mean_act - recons_harmless_mean_act
    if debug:
        assert torch.allclose((recons_harmful_acts - recons_harmless_acts).mean(0), recons_chat_refusal_dir, atol=1e-3)
    return (recons_chat_refusal_dir * chat_refusal_dir).sum(0) / (recons_chat_refusal_dir.norm() * chat_refusal_dir.norm())

records = []
for sae_name, sae in saes.items():
    print(sae_name)
    print(get_recons_refusal_dir_cos_sim(sae, chat_refusal_dir))
    records.append({
        "sae_name": sae_name,
        "cos_sim": get_recons_refusal_dir_cos_sim(sae, chat_refusal_dir).item()
    })

records_df = pd.DataFrame(records)
records_df
# %%
px.bar(
    records_df, 
    x='sae_name', 
    y='cos_sim', 
    title="Cosine similarity between true and reconstructed refusal dir"
).update_yaxes(range=[0, 1], title="Cosine sim").update_xaxes(
    ticktext=["LmSys", "Pile"],
    tickvals=records_df['sae_name'].unique(),
    title="SAE training dataset"
).show()
# %%
def get_recons_refusal_dir_mse(sae: SAE, chat_refusal_dir: torch.Tensor, debug: bool = False) -> float:
    recons_harmful_acts = sae(harmful_acts)
    recons_harmless_acts = sae(harmless_acts)

    recons_harmful_mean_act = recons_harmful_acts.mean(dim=0)
    recons_harmless_mean_act = recons_harmless_acts.mean(dim=0)
    
    recons_chat_refusal_dir = recons_harmful_mean_act - recons_harmless_mean_act
    if debug:
        assert torch.allclose((recons_harmful_acts - recons_harmless_acts).mean(0), recons_chat_refusal_dir, atol=1e-3)
    return (recons_chat_refusal_dir - chat_refusal_dir).pow(2).sum(dim=-1) / chat_refusal_dir.pow(2).sum(dim=-1)

records = []
for sae_name, sae in saes.items():
    print(sae_name)
    print(get_recons_refusal_dir_mse(sae, chat_refusal_dir))
    records.append({
        "sae_name": sae_name,
        "MSE": get_recons_refusal_dir_mse(sae, chat_refusal_dir).item()
    })

 
records_df = pd.DataFrame(records)

# %%
px.bar(
    records_df, 
    x='sae_name', y='MSE', 
    title="Relative MSE between true and reconstructed refusal dir"
).update_yaxes(range=[0, 1], title="Relative MSE").update_xaxes(
    ticktext=["LmSys", "Pile"],
    tickvals=records_df['sae_name'].unique(),
    title="SAE training dataset"
).show()
# %%
