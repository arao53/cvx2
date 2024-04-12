import torch

# a
x_a = torch.tensor([0.], requires_grad=True)
zero_a = torch.tensor([0.])
f_a = torch.pow(torch.max(x_a, zero_a),2)
f_a.backward()
print('(a) : {}'.format(x_a.grad))

# b
x_b = torch.tensor([0.], requires_grad=True)
zero_b = torch.tensor([0.])
f_b = torch.min(x_b, zero_b) + torch.max(x_b, zero_b)
f_b.backward()
print('(b) : {}'.format(x_b.grad))

# c
x_c = torch.tensor([0.], requires_grad=True)
one_c = torch.tensor([1.])
f_c = torch.min(x_c, torch.max(x_c, one_c))
f_c.backward()
print('(c) : {}'.format(x_c.grad))

# d
x_d = torch.tensor([1.], requires_grad=True)
one_d = torch.tensor([1.])
f_d = torch.min(x_d, torch.min(x_d, one_d))
f_d.backward()
print('(d) : {}'.format(x_d.grad))

# e
x_e = torch.tensor([1.], requires_grad=True)
one_e = torch.tensor([1.])
f_e = torch.min(x_e, torch.exp(torch.abs(torch.log(torch.max(x_e, one_e)))))
f_e.backward()
print('(e) : {}'.format(x_e.grad))

# f
x_f = torch.tensor([0.], requires_grad=True)
zero_f = torch.tensor([0.])
f_f = torch.min(torch.abs(x_f), x_f)
f_f.backward()
print('(f) : {}'.format(x_f.grad))