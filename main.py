from parse_args import parse_arguments
from preprocessing.load_data import build_dataloader

# MAX_LEN = 128
# TRAIN_BATCH_SIZE = 8
# VALID_BATCH_SIZE = 4
# EPOCHS = 1
# LEARNING_RATE = 1e-05




# model = BERTClass()
# model.to(device)
def main(opt):
    train_loader, val_loader, _test_loader = build_dataloader(opt)

if __name__ == '__main__':
    opt = parse_arguments()
    main(opt)
    print("----finish----")