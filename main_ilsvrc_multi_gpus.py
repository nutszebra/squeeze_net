import sys
sys.path.append('./trainer')
import argparse
import squeezenet
import data_augmentation as da
import nutszebra_ilsvrc_object_localization_with_multi_gpus
from nutszebra_optimizer import OptimizerCosineAnnealing as ilsvrc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cifar10')
    parser.add_argument('--load_model', '-m', type=str,
                        default=None,
                        help='trained model')
    parser.add_argument('--load_optimizer', '-o',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--load_log', '-l',
                        default=None,
                        help='optimizer for trained model')
    parser.add_argument('--load_data', '-ld',
                        default=None,
                        help='ilsvrc path')
    parser.add_argument('--save_path', '-p',
                        default='./',
                        help='model and optimizer will be saved at every epoch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=200,
                        help='maximum epoch')
    parser.add_argument('--batch', '-b', type=int,
                        default=32,
                        help='mini batch number')
    parser.add_argument('--gpus', '-g', nargs='+', type=int,
                        default=-1,
                        help='multiple gpu ids')
    parser.add_argument('--start_epoch', '-s', type=int,
                        default=0,
                        help='start from this epoch')
    parser.add_argument('--train_batch_divide', '-trb', type=int,
                        default=1,
                        help='divid train batch number by this')
    parser.add_argument('--test_batch_divide', '-teb', type=int,
                        default=1,
                        help='divid test batch number by this')
    parser.add_argument('--small_sample_training',
                        default=None,
                        help='If None, full dataset.')
    parser.add_argument('--parallel_train', type=int,
                        default=4,
                        help='data augmentation for training is parallelized')
    parser.add_argument('--parallel_test', type=int,
                        default=16,
                        help='data augmentation for testing is parallelized')

    args = parser.parse_args().__dict__
    if args['small_sample_training'] is not None:
        args['small_sample_training'] = int(args['small_sample_training'])
    print(args)

    print('generating model')
    model = squeezenet.Squeeze(1000)
    print('Done')
    args['model'] = model
    args['optimizer'] = ilsvrc(model=model, total_epoch=args['epoch'], start_epoch=args['start_epoch'])
    args['da'] = da.DataAugmentationNormalizeBigger
    main = nutszebra_ilsvrc_object_localization_with_multi_gpus.TrainIlsvrcObjectLocalizationClassificationWithMultiGpus(**args)
    main.run()