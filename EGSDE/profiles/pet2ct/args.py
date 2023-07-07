import argparse
argsall = argparse.Namespace(
    ckpt = 'pretrained_model/ct_diff.pt',
    dsepath = 'pretrained_model/pet2ct.pt',
    config_path = 'profiles/pet2ct/pet2ct.yml',
    t = 40,
    ls = 500,
    li = 2,
    s1 = 'cosine',
    s2 = 'neg_l2',
    phase = 'test',
    root = 'runs/',
    sample_step = 1,
    batch_size = 10,
    diffusionmodel = 'ADM',
    down_N = 16,
    seed=1234)