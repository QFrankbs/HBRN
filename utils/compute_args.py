import torch


def compute_args(args):
    # DataLoader
    if not hasattr(args, 'dataset'):  # fix for previous version
        args.dataset = 'MOSEI'
    sample_counts = [12500, 6000, 5000, 4000, 2200, 2000]  # 每个类别的样本数量
    samplecounts = [1/x for x in sample_counts]
# 计算每个类别的权重（倒数作为权重）
    weights = [count / sum(samplecounts) for count in samplecounts]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 转换为张量
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    if args.dataset == "MOSEI": args.dataloader = 'Mosei_Dataset'
    if args.dataset == "MELD": args.dataloader = 'Meld_Dataset'

    # Loss function to use
    if args.dataset == 'MOSEI' and args.task == 'sentiment': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    if args.dataset == 'MOSEI' and args.task == 'emotion': args.loss_fn = torch.nn.BCEWithLogitsLoss(weight = weights_tensor, reduction="sum")
    if args.dataset == 'MELD': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    # Answer size
    if args.dataset == 'MOSEI' and args.task == "sentiment": args.ans_size = 7
    if args.dataset == 'MOSEI' and args.task == "sentiment" and args.task_binary: args.ans_size = 2
    if args.dataset == 'MOSEI' and args.task == "emotion": args.ans_size = 6
    if args.dataset == 'MELD' and args.task == "emotion": args.ans_size = 7
    if args.dataset == 'MELD' and args.task == "sentiment": args.ans_size = 3

    if args.dataset == 'MOSEI': args.pred_func = "amax"
    if args.dataset == 'MOSEI' and args.task == "emotion": args.pred_func = "multi_label"
    if args.dataset == 'MELD': args.pred_func = "amax"

    return args
