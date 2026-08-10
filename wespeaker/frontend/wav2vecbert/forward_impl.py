import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput


def _get_model_device_dtype(encoder):
    """获取模型的设备和数据类型"""
    device = next(encoder.parameters()).device
    dtype = next(encoder.parameters()).dtype
    return device, dtype

def forward_w2v_bert(self, aud_inputs):
    """
    W2v-BERT模型的专用forward逻辑
    
    Args:
        self: AudioEncoder实例
        aud_inputs: 音频输入张量，形状为(B, T)
        
    Returns:
        BaseModelOutput: 包含last_hidden_state和hidden_states的输出
    """
    # 使用特征提取器处理音频
    features = self.feature_extractor(
        aud_inputs.cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
        truncation=False,
        return_attention_mask=False,
        )
    
    # 获取模型参数所在的设备和数据类型
    device, dtype = _get_model_device_dtype(self.upstream)
    
    # 将特征转移到正确的设备和数据类型
    input_features = features.input_features.to(device).to(dtype)
    
    # 特征投影
    x = self.upstream.feature_projection(input_features)[0]
    
    # 编码器层
    hidden_states = [x]
    for layer in self.upstream.encoder.layers:
        x = layer(x)[0]
        hidden_states.append(x)
    
    # 返回标准格式
    return BaseModelOutput(
        last_hidden_state=x,
        hidden_states=tuple(hidden_states),
        )