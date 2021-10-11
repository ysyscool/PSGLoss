def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    if epoch !=0 and epoch % decay_epoch ==0:
        decay = decay_rate ** (epoch // decay_epoch)
    else:
        decay = 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
