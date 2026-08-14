# Dependency policy

The core package keeps dependencies deliberately small: Python and PyTorch.
Dataset-specific, pretrained-model, audio, MEG, language-model, and distributed
training packages belong in optional extras after their license and platform
requirements are documented.

Do not vendor reference-repository code or restricted weights into the
Apache-2.0 core. BrainMagick's CC BY-NC material, unlicensed archives, and
gated datasets require independent implementation or a separately licensed
plugin path.
