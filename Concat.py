
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(Concat, self).__init__()
        # self.relu = nn.ReLU()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        outs = self._forward(x)
        return outs

    def _forward(self, x): # intermediate result
        if len(x) == 2:
            # w = self.relu(self.w1)
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3: # final result
            # w = self.relu(self.w2)
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
