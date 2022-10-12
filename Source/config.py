import argparse


def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train',
                        help='Model Training or Testing options')
    parser.add_argument('--dataset', type=str, default='DUTS-TR', help='DUTS')
    parser.add_argument('--data_path', type=str, default='data/')

    # Training parameter settings
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--criterion', type=str,
                        default='API', help='API or bce')
    parser.add_argument('--patience', type=int, default=3,
                        help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parser.add_argument('--pretrained', type=str, default='False')
    parser.add_argument('--pretrained_model', type=str,
                        default=None, help='pretrained model path')
    parser.add_argument('--model', type=str,
                        default='PFSNet', help='Select model (PFSNet | PFSNet_OCR | dilated_PFSNet_OCR')
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--save_map', type=str,
                        default='False', help='Save prediction map')

    # OCR parameters setting
    parser.add_argument('--in_channels', type=int, default=64)
    parser.add_argument('--key_channels', type=int, default=128)
    parser.add_argument('--mid_channels', type=int, default=256)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.05)

    # Hardware settings
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    cfg = parser.parse_args()

    return cfg


if __name__ == '__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)
