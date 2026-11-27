# 2. CodeBERT - Repository Level Split
uv run python -m software_defects train --dataset defectors --scenario Repository_Level_Split --model codebert --batch-size 4 --grad-accum 8 --gradient-checkpointing --resume
# DONE✅

# 3. CodeBERT - Within Domain ML/DL
uv run python -m software_defects train --dataset defectors --scenario Within_Domain_ML_DL --model codebert --batch-size 4 --grad-accum 8 --gradient-checkpointing --resume
# DONE✅

# 4. CodeBERT - Within Domain Non-ML
uv run python -m software_defects train --dataset defectors --scenario Within_Domain_Non_ML --model codebert --batch-size 4 --grad-accum 8 --gradient-checkpointing --resume
# DONE✅

# 5. CodeBERT - Group KFold ML/DL
uv run python -m software_defects train --dataset defectors --scenario Group_KFold_ML_DL --model codebert --batch-size 4 --grad-accum 8 --gradient-checkpointing --resume
# ONGOING 🔄️
# PAUSED 🟨

# 6. CodeBERT - Group KFold Non-ML
uv run python -m software_defects train --dataset defectors --scenario Group_KFold_Non_ML --model codebert --batch-size 4 --grad-accum 8 --gradient-checkpointing
# ONGOING 🔄️
# PAUSED 🟨

# 7. CodeT5 - ML to NonML
uv run python -m software_defects train --dataset defectors --scenario ML_to_NonML --model codet5 --batch-size 4 --grad-accum 8 --gradient-checkpointing

# 8. CodeT5 - NonML to ML
uv run python -m software_defects train --dataset defectors --scenario NonML_to_ML --model codet5 --batch-size 4 --grad-accum 8 --gradient-checkpointing

# 9. CodeT5 - Repository Level Split
uv run python -m software_defects train --dataset defectors --scenario Repository_Level_Split --model codet5 --batch-size 4 --grad-accum 8 --gradient-checkpointing

# 10. CodeT5 - Within Domain ML/DL
uv run python -m software_defects train --dataset defectors --scenario Within_Domain_ML_DL --model codet5 --batch-size 4 --grad-accum 8 --gradient-checkpointing

# 11. CodeT5 - Within Domain Non-ML
uv run python -m software_defects train --dataset defectors --scenario Within_Domain_Non_ML --model codet5 --batch-size 4 --grad-accum 8 --gradient-checkpointing

# 12. CodeT5 - Group KFold ML/DL
uv run python -m software_defects train --dataset defectors --scenario Group_KFold_ML_DL --model codet5 --batch-size 4 --grad-accum 8 --gradient-checkpointing

# 13. CodeT5 - Group KFold Non-ML
uv run python -m software_defects train --dataset defectors --scenario Group_KFold_Non_ML --model codet5 --batch-size 4 --grad-accum 8 --gradient-checkpointing