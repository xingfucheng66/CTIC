import os
import torch
import logging
from termcolor import colored
from configs import parser
from utils.engine_contrast import train, test_MAP, test
from model.contrast.model_main import MainModel
from loaders.get_loader import loader_generation
from utils.tools import fix_parameter, print_param
import utils.log_helper as log_helper
from datetime import datetime

os.makedirs('saved_model/', exist_ok=True)


def main():
    model = MainModel(args)
    device = torch.device(args.device)

    # init log
    log_helper.init_log(args)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    if not args.pre_train:
        if args.use_resnet18_cls_1000_cpt100:
            checkpoint = torch.load("/home/coder/projects/BotCL_v1/saved_model/imagenet_resnet18_cls1000_cpt100_use_slot_att.pt", map_location=device)
            model_sd = model.state_dict()
            match_sd = {k:v for k,v in checkpoint.items() if (k in model_sd) and ("cls" not in k)}
            
            model_sd.update(match_sd)
            model.load_state_dict(model_sd)

            unmatch_keys = set(model_sd.keys()) - set(match_sd.keys())
            logging.info(f"load [{len(match_sd)}/{len(model_sd)}] params \nunmatch params are {unmatch_keys}")

            fix_parameter(model, ["layer1", "layer2", "back_bone.conv1", "back_bone.bn1"], mode="fix")
            # for name, param in model.named_parameters():
            #     if 'cls' in name or 'lora' in name:
            #         param.requires_grad = True
            #         logging.info(f"only set {name} requires_grad = True")
            #     else:
            #         param.requires_grad = False
        else:
            ckpt_path = os.path.join(args.output_dir, f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            fix_parameter(model, ["layer1", "layer2", "back_bone.conv1", "back_bone.bn1"], mode="fix")
            print(colored('trainable parameter name: ', "blue"))
            print_param(model)
            logging.info(f"load {ckpt_path} pre-trained model finished, start training")
        
    else:
        print("start training the backbone")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    model.to(device)

    train_loader1, train_loader2, val_loader = loader_generation(args)
    print("data prepared")
    acc_max = 0
    print(f"batch_size = {args.batch_size}")
    for i in range(args.epoch):
        print(colored('Epoch %d/%d' % (i + 1, args.epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        if i == args.lr_drop:
            print("Adjusted learning rate to 1/10")
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
        train(args, model, device, train_loader1, optimizer, i)

        if not args.pre_train:
            map, acc = test_MAP(args, model, train_loader2, val_loader, device)
            logging.info(f"ep{i+1}-ACC: {acc}")
            # print(f"ep{i+1}-MAP: {map}")
        else:
            print("start evaluation")
            acc = test(args, model, val_loader, device)
            logging.info(f"ep{i+1}-pre_train acc is {acc}")

        if acc > acc_max:
            acc_max = acc
            print("get better result, save current model.")
            torch.save(model.state_dict(), os.path.join(args.output_dir,
                f"{args.dataset}_{args.base_model}_cls{args.num_classes}_" +
                ("transfer_" if args.use_resnet18_cls_1000_cpt100 else "") +
                f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"))

    curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('acc_log.txt', 'a') as f:
        f.write(f"max acc is {acc_max}\t\t time is {curr_time}\n")
        
if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_dir + '/', exist_ok=True)
    main()
