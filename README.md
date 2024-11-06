Code to reproduce key results accompanying "SAEs are highly dataset dependent: A case study on the refusal direction".

* [Blog Post](TODO)

# Contents

* `standard_evals.py` contains a notebook to reproduce standard SAE evals such as CE loss recovered, L0, and explained variance for the LmSys SAE evaluated on LmSys data. 
* `refusal_dir_recons_fidelity_evals.py` contains a notebook to evaluate each SAE's ability to faithfully reconstruct the refusal direction from Qwen 1.5 0.5B Chat.
* `refusal_dir_sparsity_evals.py` contains a notebook to run the sparse linear regression experiment to evaluate the sparsity of each SAE's refusal direction reconstructions.
* `refusal_latent_steering_evals.py` contains a notebook to 1) find latents in each SAE with maximum cosine similarity to the refusal direction and 2) evaluates the most aligned latents' ability to bypass refusals with the ablation technique from [Arditi et al.](https://arxiv.org/abs/2406.11717)
* `dataset_vs_model_transfer.py` contains a notebook to reproduce the heatmaps that show that the LmSys dataset is more important than the chat model for reconstructing the refusal direction.
* `train_sae.py` shows an example script to train a Qwen 1.5 0.5B Chat SAE on pile data (with instruction formatting). Run it with `python train_sae.py`
* `latent_dashboards.py` contains a notebook to generate latent dashboards for the of the latents we interpreted in the post. We use [SAE Dashboard](https://github.com/jbloomAus/SAEDashboard).

We build on code from [Arditi et al.](https://github.com/andyrdt/refusal_direction)

## Open source SAEs

We open source SAEs used in this work:

* [Qwen 1.5 0.5B Chat, LmSys](https://wandb.ai/ckkissane/qwen-500M-chat-lmsys-1m-anthropic/artifacts/model/sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768/v11/files)
* [Qwen 1.5 0.5B Chat, Pile](https://wandb.ai/ckkissane/qwen-500M-chat-pile-inst-format-with-assistant/artifacts/model/sae_qwen1.5-0.5b-chat_blocks.13.hook_resid_pre_32768/v3/files)
* [Qwen 1.5 0.5B Base, LmSys](https://wandb.ai/ckkissane/qwen-500M-base-lmsys-1m/artifacts/model/sae_qwen1.5-0.5b_blocks.13.hook_resid_pre_32768/v3/files)
* [Qwen 1.5 0.5B Base, Pile](https://wandb.ai/ckkissane/qwen-500M-base-pile-inst-format-with-assistant/artifacts/model/sae_qwen1.5-0.5b_blocks.13.hook_resid_pre_32768/v3/files)