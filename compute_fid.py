from pytorch_fid import FrechetInceptionDistance
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def get_fid_for_dir_path(path_name, start_epoch, end_epoch, iter_step, stats_path, dataset_name, model_name, if_one_dir):
    # defining stuff for FID library
    model = FrechetInceptionDistance.get_inception_model()
    model = model.cuda()
    fid = FrechetInceptionDistance(model)

    how_many_eps = len(list(range(start_epoch, end_epoch + 1)))

    result = []
    result_df = pd.DataFrame(columns=['epoch', 'dataset', 'FID_score'])

    
    if if_one_dir:
        fid_score = fid.calculate_fid_given_paths(stats_path, path_name)
        file_name = 'FID_{}_{}_{}eps.csv'.format(model_name, dataset_name, how_many_eps)
        result_df.to_csv(file_name, index=False)
    
    # measuring time
    start_time = time.time()
    for i in range(start_epoch, end_epoch + 1, iter_step):
        addon = 'epoch_' + str(i)
        path_to_dir = Path(os.path.join(path_name, addon))
        fid_score = fid.calculate_fid_given_paths(stats_path, path_to_dir)
        result_df.loc[i] = [i, dataset_name, fid_score]
        # result.append(fid_score)
        print("FID epoch {}: {}".format(i, fid_score))

    elapsed = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
    file_name = 'FID_{}_{}_{}eps.csv'.format(model_name, dataset_name, how_many_eps)
    result_df.to_csv(file_name, index=False)

if __name__ == "__main__":
    print("fid score calculator")
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=0, help='number of starting epoch')
    parser.add_argument('--end_epoch', type=int, default=100, help='Number of last epoch (inclusive).')
    parser.add_argument('--epoch_step', type=int, default=1, help='interval between epochs.')
    parser.add_argument('--images_path', type=str, required=True, help='path to dir with dirs for every epoch')
    parser.add_argument('--stats_path', type=str, required=True, help='path to file with fid statistics for given dataset')
    parser.add_argument('--dataset_name', type=str, help='name of the used dataset')
    parser.add_argument('--model_name', type=str, help='name of the used model')
    parser.add_argument('--one_dir', action='store_true', default=False, help="indicates computing fid for only one dir with imgs")
    args = parser.parse_args()

    get_fid_for_dir_path(args.images_path, args.start_epoch, args.end_epoch, args.epoch_step, args.stats_path, args.dataset_name, args.model_name, args.one_dir)