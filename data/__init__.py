from importlib import import_module
from torchvision import transforms
from utils.random_erasing import RandomErasing
from utils.random_patch import RandomPatch
from data.sampler import RandomSampler
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Data:
    def __init__(self, args):

        train_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(10),
            transforms.RandomCrop((args.height, args.width)),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            # transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            RandomPatch(prob_happen=args.random_patch_prob, patch_max_area=args.random_patch_area),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
            self.train_loader = DataLoaderX(self.trainset,
                            sampler=RandomSampler(self.trainset, args.batchid, batch_image=args.batchimage),
                            # shuffle=True,
                            batch_size=args.batchid * args.batchimage,
                            num_workers=args.nThread)
        else:
            self.train_loader = None
        
        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        else:
            raise Exception()

        self.test_loader = DataLoaderX(self.testset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = DataLoaderX(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
        
        if not args.test_only:
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images |")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d} |".format(len(self.trainset.unique_ids), len(self.trainset)))
            print("  query    | {:5d} | {:8d} |".format(len(self.queryset.unique_ids), len(self.queryset)))
            print("  gallery  | {:5d} | {:8d} |".format(len(self.testset.unique_ids), len(self.testset)))
            print("  ------------------------------")
        