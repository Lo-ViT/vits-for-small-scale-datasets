import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.build_model import create_model
from utils.dataloader import datainfo, dataload
import logging as log
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from fvcore.nn import flop_count
from pathlib import Path
from finetune import init_parser
import time
home = str(Path.home())

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    flops = 0
    wall_time = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            before = time.time()
            outputs = model(data)
            torch.cuda.synchronize() 
            wall_time += time.time() - before
            flops_dict = flop_count(model, data)
            flops += sum(flops_dict[0].values()) + sum(flops_dict[1].values())
            loss = criterion(outputs, target)
            
            test_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / total
    accuracy = 100. * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test Flops: {flops:.2f}, Test Wall Time: {wall_time:.2f}')

    return avg_loss, accuracy, flops

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    # Set up data
    data_info = datainfo(logger, args)
    _, test_dataset = dataload(args, [], [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])], data_info)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Set up model
    model = create_model(data_info['img_size'], data_info['n_classes'], args).to(device)
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location=device)["model_state_dict"])

    # Test model
    test_model(model, test_loader, criterion, device)

if __name__ == "__main__":
    parser = init_parser()
    # Add arguments similar to those in the main training script
    parser.add_argument('--model', type=str, help='Path to the trained model weights')
    # logger

    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.setLevel(level=log.DEBUG)
    
    # Include other necessary arguments
    args = parser.parse_args()
    print(args.model)
    main(args)
