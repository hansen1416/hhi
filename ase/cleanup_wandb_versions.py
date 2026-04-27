import wandb
import sys

entity = "yugoamaryl"
project = "hhi" 
artifact_name = "hhi_film_model"
artifact_name = "hhi_film_transfer"
type_name = "model"

api = wandb.Api()

full_name = f"{entity}/{project}/{artifact_name}"
print(f"🔍 Fetching ALL versions of artifact: {full_name}")

versions = list(api.artifacts(name=full_name, type_name=type_name))

if not versions:
    print("No versions found.")
    sys.exit(0)

print(f"Found {len(versions)} versions. Deleting...")

for v in versions:
    print(f"  → Deleting {v.name} ({v.version})")
    v.delete(delete_aliases=True)

print(f"✅ All versions of {artifact_name} deleted.")
print("   The artifact itself remains (empty) in your WandB UI.")
print("   Ready for fresh uploads from your HUMOS-conditioned PHC training.")