# Target-free inference contract

`forward(..., labels=...)` is for supervised loss computation only. Public text generation must call a target-free method whose inputs contain neural tensors, permitted neural metadata, and explicit decoding configuration only.

Every generated prediction records the decoder mode, neural representation, alignment regime, control condition, random seed, and token trace. The test suite must prove that omitted, permuted, or replaced labels cannot change generation output.
