import torch
from torch.autograd import Function, Variable

def soft_dice_coef_batch_voluem(input, target):
    '''
    consider batch as voluem, and compute dice across whole batch samples as one,
    but class wise, after that per class dices average performed.
    '''
    smooth = 0.0001
    assert len(input.size()) == 4 #batch, class, h, w
    assert len(target.size()) == 4
    class_num = target.size(1)
    input = input.transpose(0, 1).contiguous() #flip batch and classes dimentions
    target = target.transpose(0, 1).contiguous()
    input_flat = input.view(class_num, -1)
    target_flat = target.view(class_num, -1)
    inter = torch.sum(input_flat * target_flat, -1)
    union = input_flat.sum(-1) + target_flat.sum(-1)
    assert union.size(0) == class_num
    assert inter.size(0) == class_num
    soft_dice = torch.mean((2*inter + smooth) / (union + smooth)) #mean acros classes
    return soft_dice

def dice_loss(input, target):
    soft_dice = soft_dice_coef_batch_voluem(input, target)
    return -2 * soft_dice


def dice_coef_for_val(input, target):
    """Dice coeffs (class wise) for evaluation, should be implemented
    subject wise. Batch should contain only one subject. Because of huge batch size
    computation performed on cpu"""
    assert len(input.size()) == 4
    assert len(target.size()) == 4
    input = input.transpose(0, 1).contiguous() #flip batch and classes dimentions
    target = target.transpose(0, 1).contiguous()
    dices = []
    for i, t in zip(input, target):
        if torch.all(i == 0) and torch.all(t == 0):
        #prediction and target is empty
            dices.append(-1)
            continue
        input_flat = i.view(-1)
        target_flat = t.view(-1)
        inter = torch.sum(input_flat * target_flat)
        union = input_flat.sum() + target_flat.sum()
        assert union != 0
        dice = 2*inter/union
        dices.append(dice.item())
    return dices
