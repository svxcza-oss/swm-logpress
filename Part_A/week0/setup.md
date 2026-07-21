\# A Track Environment Setup



\## Local Desktop Test



\- OS: Windows

\- Python: 3.11.9

\- CUDA available: True

\- Model: Qwen/Qwen2.5-3B-Instruct

\- Test: short log input + past\_key\_values access

\- Result: Success



\## Verified



\- torch import: success

\- transformers import: success

\- kvpress import: success

\- model load: success

\- forward pass: success

\- KV cache shape output: success



\## Note



This environment is for local development and small-model KV cache tests.

Official experiment environment should be pinned again on RunPod/RTX 4090 after 7B\~8B model and baseline reproduction succeed.





\- GPU: RTX 5060 Ti

\- VRAM: 16GB



num layers: 36

layer0 key shape: torch.Size(\[1, 2, 48, 128])

layer0 value shape: torch.Size(\[1, 2, 48, 128])

input token count: 48

max memory GB: 5.772336959838867

