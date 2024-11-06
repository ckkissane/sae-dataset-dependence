# %%
from utils import *
from colorama import Fore
import textwrap
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
def get_latent_cosine_sims(sae: SAE, chat_refusal_dir: torch.Tensor) -> torch.Tensor:
    cosine_sims = (sae.W_dec @ chat_refusal_dir) / (sae.W_dec.norm(dim=1) * chat_refusal_dir.norm())
    return cosine_sims

records = []

for sae_name, sae in saes.items():
    print(sae_name)
    cosine_sims = get_latent_cosine_sims(sae, chat_refusal_dir)
    print(cosine_sims.topk(5))
    
    records.append({
        "sae_name": sae_name,
        "max_cos_sim": cosine_sims.max().item(),
    })
    print()
    
records_df = pd.DataFrame(records)
filtered_df = records_df[records_df['sae_name'].str.contains("32768")]
px.bar(
    filtered_df, 
    x='sae_name', 
    y='max_cos_sim', 
    title="Max cosine sim between latents and true refusal direction",
    ).update_yaxes(range=[0, 1], title="Max cosine sim").update_xaxes(
    ticktext=["LmSys", "Pile"],
    tickvals=filtered_df['sae_name'].unique(),
    title="SAE training dataset"
).show()
# %%
# Bypass refusal harmful
def bypass_harmful_refusal(sae: SAE, latent_id: int, n_inst_test: int, max_tokens_generated: int = 16, batch_size: int = 16,):
    inst_test = harmful_inst_test
    intervention_layers = list(range(model.cfg.n_layers))

    sae_refusal_feature = sae.W_dec[latent_id]
    
    recons_chat_hook_fn = functools.partial(direction_ablation_hook, direction=sae_refusal_feature)
    recons_chat_fwd_hooks = [(utils.get_act_name(act_name, l), recons_chat_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
    recons_chat_intervention_generations = get_generations(model, inst_test[:n_inst_test], tokenize_instructions_fn, fwd_hooks=recons_chat_fwd_hooks, max_tokens_generated=max_tokens_generated, batch_size=batch_size)


    return recons_chat_intervention_generations
    
sae_to_refusal_latent = {
    "chat_lmsys_32768": 25840,
    "chat_pile_inst_format_with_assistant_32768": 25271
}

records = []
n_inst_test = 100 # arg
max_tokens_generated = 64 # arg
batch_size = 16 # arg

for sae_name in sae_to_refusal_latent.keys():
    print(sae_name)
    sae = saes[sae_name]
    latent_id = sae_to_refusal_latent[sae_name]
    completions = bypass_harmful_refusal(sae, latent_id, n_inst_test=n_inst_test, max_tokens_generated=max_tokens_generated, batch_size=batch_size,)

    refusal_score = get_refusal_scores(completions)
    records.append({
        "sae_name": sae_name,
        "latent_id": latent_id,
        "completions": completions,
        "refusal_score": refusal_score
    })
# %%
print("bypassing refusal on harmful with true refusal dir")
n_inst_test = 100 # arg
max_tokens_generated = 64 # arg
batch_size = 16 # arg

inst_test = harmful_inst_test

intervention_layers = list(range(model.cfg.n_layers))

chat_hook_fn = functools.partial(direction_ablation_hook, direction=chat_refusal_dir)

print(f"Generating completions ablating chat refusal vector (harmful)")
chat_fwd_hooks = [(utils.get_act_name(act_name, l), chat_hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
chat_intervention_generations = get_generations(model, inst_test[:n_inst_test], tokenize_instructions_fn, fwd_hooks=chat_fwd_hooks, max_tokens_generated=max_tokens_generated, batch_size=batch_size)
    
records.append({
    "sae_name": "chat_refusal_dir",
    "latent_id": None,
    "completions": chat_intervention_generations,
    "refusal_score": get_refusal_scores(chat_intervention_generations)
})
# %%
print("baseline refusal on harmful with no intervention")
n_inst_test = 100 # arg
max_tokens_generated = 64 # arg
batch_size = 16 # arg

inst_test = harmful_inst_test


print(f"Generating completions ablating chat refusal vector (harmful)")
baseline_generations = get_generations(model, inst_test[:n_inst_test], tokenize_instructions_fn, fwd_hooks=[], max_tokens_generated=max_tokens_generated, batch_size=batch_size)

for i in range(n_inst_test):
    print(f"INSTRUCTION {i}: {repr(inst_test[i])}")
    print(Fore.RED + f"INTERVENTION COMPLETION:")
    print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)
    
records.append({
    "sae_name": "baseline",
    "latent_id": None,
    "completions": baseline_generations,
    "refusal_score": get_refusal_scores(baseline_generations)
})
# %%
records_df = pd.DataFrame(records)
records_df
# %%
filtered_df = records_df[records_df['sae_name'] != "chat_refusal_dir"]
filtered_df = filtered_df[records_df['sae_name'] != "baseline"]
fig = px.bar(
    filtered_df, 
    x='sae_name', y='refusal_score', 
    title="Refusal score on harmful inst with top refusal latent ablated"
)

fig.add_hline(y=get_refusal_scores(chat_intervention_generations), 
              line_dash="dot", 
              line_color="red", 
              annotation_text="full steering vector", 
              annotation_position="right",
              annotation_x=1.0,
              annotation_xanchor="left")

fig.add_hline(y=get_refusal_scores(baseline_generations), 
              line_dash="dot", 
              line_color="green", 
              annotation_text="baseline", 
              annotation_position="right",
              annotation_x=1.0,
              annotation_xanchor="left")

fig.update_layout(
    yaxis_range=[0, 1],  # Set y-axis range from 0 to 1
    annotations=[
        dict(
            x=1.0,
            y=get_refusal_scores(chat_intervention_generations),
            xref="paper",
            yref="y",
            text="refusal dir",
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(color="red")
        ),
        dict(
            x=1.0,
            y=get_refusal_scores(baseline_generations),
            xref="paper",
            yref="y",
            text="baseline",
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(color="green")
        )
    ]
)

fig.update_yaxes(title="Refusal score").update_xaxes(
    ticktext=["LmSys (latent 25840)", "Pile (latent 25271)"],
    tickvals=filtered_df['sae_name'].unique(),
    title="SAE training dataset"
).show()
# %%
# print examples
def print_completions(baseline_completions, lmsys_sae_completions, pile_sae_completions, steering_completions, inst_test):
    for i in range(len(lmsys_sae_completions)):
        print(f"INSTRUCTION: {repr(inst_test[i])}")
        print(Fore.GREEN + f"BASELINE COMPLETION:")
        print(textwrap.fill(repr(baseline_completions[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)
        print(Fore.RED + f"REFUSAL DIR INTERVENTION COMPLETION:")
        print(textwrap.fill(repr(steering_completions[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)
        print(Fore.BLUE + f"LMSYS SAE INTERVENTION COMPLETION:")
        print(textwrap.fill(repr(lmsys_sae_completions[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)

        print(Fore.MAGENTA + f"PILE SAE INTERVENTION COMPLETION:")
        print(textwrap.fill(repr(pile_sae_completions[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
        print(Fore.RESET)

num_to_print = 100

baseline_completions = records_df[records_df['sae_name'] == "baseline"]["completions"].iloc[0]
refusal_dir_compleitions = records_df[records_df['sae_name'] == "chat_refusal_dir"]["completions"].iloc[0]
lmsys_sae_completions = records_df[records_df['sae_name'] == "chat_lmsys_32768"]["completions"].iloc[0]
pile_sae_completions = records_df[records_df['sae_name'] == "chat_pile_inst_format_with_assistant_32768"]["completions"].iloc[0]

print_completions(baseline_completions[:num_to_print], lmsys_sae_completions[:num_to_print], pile_sae_completions[:num_to_print], refusal_dir_compleitions[:num_to_print], harmful_inst_test[:num_to_print])
# %%