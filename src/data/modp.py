import torch

def split_data(data, split_ratio=0.5):
    perm = torch.randperm(data.shape[1])
    train_idx = perm[:int(data.shape[1] * split_ratio)]
    valid_idx = perm[int(data.shape[1] * split_ratio):]
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]
    print(f"Train data: {train_data.shape}")
    print(f"Valid data: {valid_data.shape}")
    return train_data, valid_data

# Compute y^exp (mod p)
# y: tensor
# exp, p: int
def mod_exp(y, exp, p):
    result = torch.ones_like(y)
    base = y % p
    rem = exp
    while rem > 0:
        if rem % 2 == 1:
            result = (result * base) % p
        base = (base * base) % p
        rem = rem // 2
    return result

def mod_p_data(p, eq_token, op_token, task="multiplication", device="cuda"): # TODO: undo device="cuda"
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """

    x = torch.arange(p, device=device)
    if task == "division" or task == "parity_division":
        y = torch.arange(1, p, device=device)
    else:
        y = torch.arange(p, device=device)
    x, y = torch.cartesian_prod(x, y).T

    if task == "multiplication":
        result = (x * y) % p
    elif task == "addition":
        result = (x + y) % p
    elif task == "subtraction":
        result = (x - y) % p
    elif task == "division":
        result = (x * mod_exp(y, p-2, p)) % p
    elif task == "parity_division":
        mask = (y % 2) != 0
        y_inv = mod_exp(y, p - 2, p)
        div_result = (x * y_inv) % p
        sub_result = (x - y) % p
        result = torch.where(mask, div_result, sub_result)
    elif task == "sum_of_squares":
        result = (x**2 + y**2) % p
    elif task == "quad1":
        result = (x**2 + x*y + y**2) % p
    elif task == "quad2":
        result = (x**2 + x*y + y**2 + x) % p
    elif task == "cubic1":
        result = (x**3 + x*y) % p
    elif task == "cubic2":
        result = (x**3 + x*(y**2) + y) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"

    eq = torch.ones_like(x, device=device) * eq_token
    op = torch.ones_like(x, device=device) * op_token

    return torch.stack([x, op, y, eq, result])