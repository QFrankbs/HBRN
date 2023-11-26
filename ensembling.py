import argparse, os, glob, pickle, warnings, torch
import numpy as np
#from utils.pred_func import *
from sklearn.metrics import classification_report
from utils.compute_args import compute_args
from torch.utils.data import DataLoader
from mosei_dataset import Mosei_Dataset
from meld_dataset import Meld_Dataset
#from model_LA import Model_LA
#from model_LAV import Model_LAV
from new_model_12 import Model_LAV_RETNET
from new_train import evaluate
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")

def amax(x):
    return np.argmax(x, axis=1)

def multi_label(x):
    return (x > 0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--sets',  nargs='+', default=["valid", "test"])

    parser.add_argument('--index', type=int, default=99)
    parser.add_argument('--private_set', type=str, default=None)

    args = parser.parse_args()
    return args
def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


if __name__ == '__main__':
    args = parse_args()

    # Save vars
    private_set = args.private_set
    index = args.index
    sets = args.sets

    # Listing sorted checkpoints
    ckpts = sorted(glob.glob(os.path.join(args.output, args.name,'best*')), reverse=True)

    # Load original args
    args = torch.load(ckpts[0])['args']
    args = compute_args(args)


    # Define the splits to be evaluated
    evaluation_sets = list(sets) + ([private_set] if private_set is not None else [])
    print("Evaluated sets: ", str(evaluation_sets))
    # Creating dataloader
    train_dset = eval(args.dataloader)('train', args)
    loaders = {setin: DataLoader(eval(args.dataloader)(setin, args),
               args.batch_size,
               num_workers=8,
               pin_memory=True) for setin in evaluation_sets}

    # Creating net
    net = eval(args.model)(args).cuda()

    # Ensembling sets
    ensemble_preds = {setin: {} for setin in evaluation_sets}
    ensemble_accuracies = {setin: [] for setin in evaluation_sets}

    # Iterating over checkpoints
    for i, ckpt in enumerate(ckpts):

        if i >= index:
            break

        print("###### Ensembling " + str(i+1))
        state_dict = torch.load(ckpt)['state_dict']
        net.load_state_dict(state_dict)

        # Evaluation per checkpoint predictions
        for setin in evaluation_sets:
            accuracy, preds = evaluate(net, loaders[setin], args)
            print('Accuracy for ' + setin + ' for model ' + ckpt + ":", accuracy)
            for idx, pred in preds.items():
                if idx not in ensemble_preds[setin]:
                    ensemble_preds[setin][int(idx.item())] = []
                ensemble_preds[setin][int(idx.item())].append(pred)

            # Compute set ensembling accuracy
            # Get all ids and answers
            ids = [int(idx.item()) for ids, _, _, _, _, _ in loaders[setin] for idx in ids]
            # print(ids)
            #print(evaluation_sets)
            ans = [np.array(a) for _, _, _, _, _, ans in loaders[setin] for a in ans]
            print(type(ans[0]))
            # print(ensemble_preds[set])
            #print(ensemble_preds[setin][idx],idx,setin)
            # for all id, get averaged probabilities
            avg_preds = np.array([np.mean(np.array(ensemble_preds[setin][idx]), axis=0) for idx in ids])
            # Compute accuracies
            if setin != private_set:
                if args.task == "emotion":
                    pred_re = multi_label(avg_preds)
                    accuracy = np.mean(pred_re == ans) * 100
                    print(type(pred_re),pred_re.shape)
                    ans = np.array(ans)
                    

                   # Example predicted and actual results (replace these with your actual data)

                    # Calculate accuracy and F1-score for each emotion column
                    accuracies = []
                    f1_scores = []
                    for col in range(ans.shape[1]):
                        accuracy = accuracy_score(ans[:, col], pred_re[:, col])
                        f1 = f1_score(ans[:, col], pred_re[:, col])
                        accuracies.append(accuracy)
                        f1_scores.append(f1)

                    # Print results
                    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
                        print(f"Emotion {i+1}: Accuracy = {acc:.4f}, F1-score = {f1:.4f}")             
                    
                else:
                    pred_re = amax(avg_preds)
                    accuracy = np.mean(pred_re == ans) * 100

                ensemble_accuracies[setin].append(accuracy)

            print(classification_report(ans, eval(args.pred_func)(avg_preds)))

    # Printing overall results
    for setin in sets:
        print("Max ensemble w-accuracies for " + setin + " : " + str(max(ensemble_accuracies[setin])))


