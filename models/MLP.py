import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    def __init__(self,num_frames:int, num_actions: int):
        super(MLPNetwork, self).__init__()
        
        input_dim = num_frames * 128 
        # Định nghĩa kiến trúc mạng như bạn yêu cầu
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),       # Đầu vào: 128 bytes RAM
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions) # Đầu ra: Q-values hoặc Logits của các hành động
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bắt buộc ép kiểu về float32 vì dữ liệu gốc từ Gymnasium thường là uint8
        x = x.to(torch.float32)
        
        # Chuẩn hóa dữ liệu từ [0, 255] về đoạn [0.0, 1.0] 
        # Bước này cực kỳ quan trọng giúp mạng Neural học nhanh và ổn định hơn
        x = x / 255.0
        
        # Đưa qua mạng neural và trả về kết quả
        return self.network(x)