import os, torch, torch.distributed as dist
import mbridge.core.llm_bridge as lb
_o=lb.get_gpt_decoder_block_spec
def _p(c,**k): k["use_transformer_engine"]=False; return _o(c,**k)
_p.__signature__=__import__("inspect").signature(_o); lb.get_gpt_decoder_block_spec=_p
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from transformers import Qwen2Config
from mbridge import AutoBridge
torch.cuda.set_device(0); dist.init_process_group("nccl")
mpu.initialize_model_parallel(tensor_model_parallel_size=1); model_parallel_cuda_manual_seed(1234)
cfg=Qwen2Config(hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,vocab_size=128,max_position_embeddings=128)
b=AutoBridge.from_config(cfg)
for _c in ([b.config] if hasattr(b,"config") else []):
    try:_c.sequence_parallel=False
    except:pass
m=b.get_model(weight_path=None)[0]
l2g=b._weight_name_mapping_mcore_local_to_global(m)
print("named_parameters (first 4):", [n for n,_ in list(m.named_parameters())[:4]])
print("l2g keys (first 4):", list(l2g.keys())[:4])
print("l2g sample val:", list(l2g.items())[:2])
