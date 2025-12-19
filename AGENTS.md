# Repository Guidelines

## Project Structure & Module Organization
- `pose/` hosts the semi-supervised pose stack built on MMEngine + MMPose; key modules live under `pose/semisup/` (current wrapper, utils) and `pose/configs/` (training recipes).  
- Legacy references are kept in folders suffixed with `_old`; consult them for context but implement changes in the active modules.  
- Core Sapiens pretraining code resides in `pretrain/`, while other perception tasks (`det/`, `seg/`, `cv/`) share utilities but are not part of the SAGE-Pose flow.  

## Build, Test, and Development Commands
- Install pose dependencies (inside a fresh virtual environment):  
  `pip install -r pose/requirements.txt`  
- Launch semi-supervised training (single GPU example):  
  `python pose/tools/train.py pose/configs/sapiens_pose/semisup/stage1_sagepose_semi.py --work-dir work_dirs/sagepose_stage1`  
- Resume or tweak configs inline using `--cfg-options key=value`; export `PYTHONPATH="/home/xiangxiantong/sapiens:/home/xiangxiantong/sapiens/pretrain:$PYTHONPATH"` before running any MMEngine command.

## Coding Style & Naming Conventions
- Python code follows Black-ish 4-space indentation; keep imports grouped (stdlib, third party, local).  
- Prefer explicit names (`geom_teacher`, `lambda_u_eff`) and snake_case for variables, files, and directories.  
- Align with existing docstrings and comments: concise, task-focused notes only.  
- Run `ruff` or `flake8` locally if editing shared utilities; add type hints when they clarify tensor shapes.

## Testing Guidelines
- Fast unit checks live under `pose/tests/` (pytest). Run `pytest pose/tests -q` before opening a PR that touches Python logic.  
- Integration smoke tests mirror training scripts; dry-run with a tiny dataset or `--max-epochs 1` to validate pipelines when changing dataloaders or configs.  
- Attach logs or tensorboard snippets when reporting regressions in semi-supervised routines.

## Commit & Pull Request Guidelines
- Commit messages should be short, imperative clauses (`Fix EMA warmup`, `Add KL mask logging`), mirroring existing history.  
- Squash WIP commits locally; each PR should describe the intent, highlight affected configs or scripts, and link to data sources or issues.  
- Provide reproduction details (command, config overrides, dataset slice). Include before/after metrics when altering training behaviour or losses.  
- Request reviews from maintainers familiar with `pose/semisup/`; flag breaking API changes in the summary and update dependent configs simultaneously.

## Environment & Security Tips
- Never commit checkpoints or raw datasets; reference absolute paths via config variables (`DATA_ROOT`).  
- Store secrets (API keys, private endpoints) outside the repo; pass them through environment variables when scripting automation.  
- When running distributed jobs, align NCCL settings with the guidance in `SAGE-Pose（基于 Sapiens 的半监督微调）一页说明.md` to avoid communication stalls.
