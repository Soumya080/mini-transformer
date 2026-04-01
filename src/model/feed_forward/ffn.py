import torch 
import torch.nn  as nn
from torch.autograd import Function

# from model.feed_forward.ffn import FeedForward

class FFN(Function):
    @staticmethod
    def forward(ctx,X,w1,b1,w2,b2):
        h = X @ w1 + b1
        a = torch.relu(h)
        y = a @ w2 + b2
        
        ctx.save_for_backward(X,h,a,w1,w2)
        
        return y
    @staticmethod
    def backward(ctx,dy):
        X,h,a,w1,w2 = ctx.saved_tensors
        
        B, T,_ = X.shape
        
        dW2 = a.reshape(-1, a.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
        db2 = dy.sum(dim=(0, 1))

        da = dy @ w2.T

        # --------------------
        # Backprop ReLU
        # --------------------
        dh = da * (h > 0)

        # --------------------
        # Backprop Linear 1
        # --------------------
        dW1 = X.reshape(-1, X.shape[-1]).T @ dh.reshape(-1, dh.shape[-1])
        db1 = dh.sum(dim=(0, 1))

        dx = dh @ w1.T

        # Return gradients in the SAME order as forward inputs
        return dx, dW1, db1, dW2, db2
    
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ffn):
        super().__init__()     
        self.w1 = nn.Parameter(torch.rand(d_model,d_ffn)*0.02)
        self.b1 = nn.Parameter(torch.zeros(d_ffn))
            
        self.w2 = nn.Parameter(torch.rand(d_ffn,d_model)*0.02)
        self.b2 = nn.Parameter(torch.zeros(d_model))
            
    def forward(self,X):  
        return FFN.apply(X, self.w1,self.b1,self.w2,self.b2)
        
        
if __name__ == "__main__":
    
    B, T = 2, 4
    d_model = 8
    d_ff = 16

    x = torch.randn(B, T, d_model, requires_grad=True)

    ffn = FeedForward(d_model, d_ff)

    out = ffn(x)
    loss = out.sum()

    loss.backward()
    
    print("✅ Forward output shape:", out.shape)
    print("✅ x.grad shape:", x.grad.shape)
    print("✅ W1.grad shape:", ffn.w1.grad.shape)
    print("✅ W2.grad shape:", ffn.w2.grad.shape)
    print("🎉 FFN forward + backward works correctly")
    # print("✅ Forwar

