FROM hansen1416/phc:v1

# Install the two packages you specifically requested
# (pydantic pin for RL-Games / ModelPHCContinuous compatibility + wandb for clean training logs)
RUN python -m pip install --no-cache-dir -U "pydantic>=1.10.8,<2" && \
    python -m pip install --no-cache-dir -U "wandb==0.22.3"

# Keep the container alive forever — required for RunPod + long hhi training runs
# (ASE/PHC + HUMOS 128 body-shape/gender variations → physical motion converter)
CMD ["tail", "-f", "/dev/null"]