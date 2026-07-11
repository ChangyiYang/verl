import os, torch, torch.distributed as dist
from megatron.core import parallel_state as mpu
import mbridge.core.llm_bridge as lb
# force local (non-TE) layer spec: param layout is identical, avoids missing transformer_engine
_orig = lb.get_gpt_decoder_block_spec
def _patched(config, **kw):
    kw["use_transformer_engine"] = False
    return _orig(config, **kw)
_patched.__signature__ = __import__("inspect").signature(_orig)  # keep vp_stage detection
lb.get_gpt_decoder_block_spec = _patched

from transformers import Qwen2Config
from mbridge import AutoBridge

import os as _os; _lr=int(_os.environ.get("LOCAL_RANK",0)); torch.cuda.set_device(_lr); dist.init_process_group("nccl")
mpu.initialize_model_parallel(tensor_model_parallel_size=int(os.environ.get("TP","1")))
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)
torch.manual_seed(0)
cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, vocab_size=128, max_position_embeddings=128)
bridge = AutoBridge.from_config(cfg)
print("bridge:", type(bridge).__name__)
models = bridge.get_model(weight_path=None)
m = models[0]
print("n_params:", sum(1 for _ in m.named_parameters()))
for i,(n,p) in enumerate(m.named_parameters()):
    print(f"  {n}  shape={tuple(p.shape)} tp={getattr(p,'tensor_model_parallel',False)} pdim={getattr(p,'partition_dim',None)}")
    if i>=10: break
print("--- export_weights first few ---")
for i,(hn,ht) in enumerate(bridge.export_weights(models)):
    print(f"  {hn}  shape={tuple(ht.shape)}")
    if i>=6: break
print("PROBE OK")
