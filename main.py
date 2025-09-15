from models.sam import sam_model_registry
import numpy as np
import os
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from utils.dataset import Public_dataset
from pathlib import Path
from utils.metrics import compute_dice, compute_nsd
import monai
import cfg
import json

# Use the arguments
args = cfg.parse_args()
args.num_cls = 1
args.epochs = 3000
print(args)

def train_model(trainloader,valloader,testloader,dir_checkpoint,epochs):
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
   
    if args.model == 'random':
        sam = sam_model_registry[args.arch](args,checkpoint=None,num_classes=args.num_cls, image_size=args.image_size)
    if args.model == 'sam':
        sam = sam_model_registry[args.arch](args,checkpoint=args.sam_ckpt,num_classes=args.num_cls, image_size=args.image_size)
    if args.model == 'medsam':
        sam = sam_model_registry[args.arch](args,checkpoint="/data/humanBodyProject/mri_foundation_model/pretrained_weights/medsam_vit_b.pth", num_classes=args.num_cls, image_size=args.image_size)
    if args.model == 'ours':
        sam = sam_model_registry[args.arch](args,checkpoint="/data/humanBodyProject/mri_foundation_model/dinov2/dino_vitb+sam_0429_nolayerscale_smallerlr/eval/training_47535/teacher_checkpoint.pth", num_classes=args.num_cls, image_size=args.image_size, pretrained_sam=True)

    if args.finetune_type == 'adapter':
        for n, value in sam.named_parameters():
            if "Adapter" not in n: # only update parameters in adapter
                value.requires_grad = False
        print('if update encoder:',args.if_update_encoder)
        print('if image encoder adapter:',args.if_encoder_adapter)
        print('if mask decoder adapter:',args.if_mask_decoder_adapter)
        if args.if_encoder_adapter:
            print('added adapter layers:',args.encoder_adapter_depths)
    elif args.finetune_type == 'vanilla' and args.if_update_encoder==False:   
        print('if update encoder:',args.if_update_encoder)
        for n, value in sam.image_encoder.named_parameters():
            value.requires_grad = False
    elif args.finetune_type == 'lora':
        print('if update encoder:',args.if_update_encoder)
        print('if image encoder lora:',args.if_encoder_lora_layer)
        print('if mask decoder lora:',args.if_decoder_lora_layer)
        sam = LoRA_Sam(args,sam,r=4).sam
    sam.to('cuda')
        
    optimizer = optim.AdamW(sam.parameters(), lr=b_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    optimizer.zero_grad()
    
    criterion = monai.losses.DiceCELoss(sigmoid=True)

    iter_num = 0
    max_iterations = epochs * len(trainloader) 
    
    val_largest_dsc = 0
    last_update_epoch = 0

    for epoch in range(epochs):
        sam.train()

        for i,data in enumerate(trainloader):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
            msks = msks.cuda()

            if args.if_update_encoder:
                img_emb = sam.image_encoder(imgs)
            else:
                with torch.no_grad():
                    img_emb = sam.image_encoder(imgs)
            
            # get default embeddings
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, _ = sam.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
            
            loss = criterion(pred, msks)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            else:
                if args.if_warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, 'Shift iter is %s, smaller than zero' % shift_iter
                    lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

            iter_num+=1

        if epoch % 10 == 0:
            eval_loss = 0
            sam.eval()
            name2pred = {}
            name2mask = {}
            with torch.no_grad():
                for i,data in enumerate(valloader):
                    imgs = data['image'].cuda()
                    msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
                    msks = msks.cuda()

                    img_emb = sam.image_encoder(imgs)
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = sam.mask_decoder(
                                    image_embeddings=img_emb,
                                    image_pe=sam.prompt_encoder.get_dense_pe(), 
                                    sparse_prompt_embeddings=sparse_emb,
                                    dense_prompt_embeddings=dense_emb, 
                                    multimask_output=True,
                                  )
                    loss = criterion(pred, msks)
                    eval_loss += loss.item()

                    prob = (pred > 0).cpu().long()
                    mask = msks.cpu().long()

                    for idx, name in enumerate(data['patient_name']):
                        if name not in name2pred:
                            name2pred[name] = [prob[idx].unsqueeze(0)]
                            name2mask[name] = [mask[idx].unsqueeze(0)]
                        else:
                            name2pred[name].append(prob[idx].unsqueeze(0))
                            name2mask[name].append(mask[idx].unsqueeze(0))

            eval_loss /= (i+1)

            val_dsc_3d = []
            for name in name2pred:
                pred_list = name2pred[name]
                mask_list = name2mask[name]
                pred_3d = torch.cat(pred_list, 0).squeeze()
                mask_3d = torch.cat(mask_list, 0).squeeze()
                
                tmp_dsc = compute_dice(pred_3d, mask_3d)
                val_dsc_3d.append(tmp_dsc)

            val_dsc_3d = np.mean(np.array(val_dsc_3d))

            if val_dsc_3d > val_largest_dsc:
                val_largest_dsc = val_dsc_3d
                last_update_epoch = epoch 
                
                torch.save(sam.state_dict(), 'temp/' + args.dataset_name + '_' + args.model + '_best.pth')

            elif (epoch-last_update_epoch)>=200 and epoch >= 1000:
                print('Training finished###########')
                break

            print('Eval Epoch num %s | val loss %.4f | dsc %.4f | best %.4f' % (epoch,eval_loss,val_dsc_3d,val_largest_dsc))
            
    load_path = 'temp/' + args.dataset_name + '_' + args.model + '_best.pth'
    print('Load from', load_path)
    sam.load_state_dict(torch.load(load_path))
    sam.eval()
    
    name2pred = {}
    name2mask = {}
    
    test_dsc_2d = []
    with torch.no_grad():
        for i,data in enumerate(testloader):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask'])
            msks = msks.cuda()

            img_emb = sam.image_encoder(imgs)
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, _ = sam.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=sam.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                        )
           
            prob = (pred > 0).cpu().long()
            mask = msks.cpu().long()
            
            tmp_dsc = compute_dice(prob, mask)
            test_dsc_2d.append(tmp_dsc)

            for idx, name in enumerate(data['patient_name']):
                if name not in name2pred:
                    name2pred[name] = [prob[idx].unsqueeze(0)]
                    name2mask[name] = [mask[idx].unsqueeze(0)]
                else:
                    name2pred[name].append(prob[idx].unsqueeze(0))
                    name2mask[name].append(mask[idx].unsqueeze(0))

    test_dsc_3d = []
    test_nsd_3d = []
    for name in name2pred:
        pred_list = name2pred[name]
        mask_list = name2mask[name]

        pred_3d = torch.cat(pred_list, 0).squeeze()
        mask_3d = torch.cat(mask_list, 0).squeeze()

        tmp_dsc = compute_dice(pred_3d, mask_3d)
        tmp_nsd = compute_nsd(pred_3d, mask_3d)
        test_dsc_3d.append(tmp_dsc)
        test_nsd_3d.append(tmp_nsd)

        print(name, tmp_dsc, tmp_nsd, pred_3d.shape)
    
    test_dsc_2d = np.mean(test_dsc_2d)
    test_dsc_3d = np.mean(test_dsc_3d)
    test_nsd_3d = np.mean(test_nsd_3d)
    print(test_dsc_2d, test_dsc_3d, test_nsd_3d)

    return test_dsc_2d, test_dsc_3d, test_nsd_3d
                
                
                
if __name__ == "__main__":
    dataset_name = args.dataset_name
    train_img_list = args.train_img_list
    val_img_list = args.val_img_list
    test_img_list = args.test_img_list
    
    num_workers = 0
    Path(args.dir_checkpoint).mkdir(parents=True,exist_ok=True)
    path_to_json = os.path.join(args.dir_checkpoint, "args.json")
    args_dict = vars(args)
    with open(path_to_json, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

    n_type = args.model
    args.n_type = n_type

    args.if_spatial = True
    args.b = 2
        
    delete_empty_masks = True
    test_dataset = Public_dataset(args,args.img_folder, args.mask_folder, test_img_list, phase='test',targets=[args.targets],normalize_type=n_type,if_prompt=False,crop_size=args.image_size, delete_empty_masks=delete_empty_masks, target_cls=args.cls)
    testloader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=8)

    
    final_score_list = []
    for repeat in range(10):
        print('Running', n_type, args, repeat)
        train_dataset = Public_dataset(args,args.img_folder, args.mask_folder, train_img_list, phase='train',targets=[args.targets],normalize_type=n_type,if_prompt=False,crop_size=args.image_size, few_shot=True, seed=repeat, delete_empty_masks=delete_empty_masks, target_cls=args.cls, if_spatial=args.if_spatial)
        val_dataset = Public_dataset(args,args.img_folder, args.mask_folder, val_img_list, phase='val',targets=[args.targets],normalize_type=n_type,if_prompt=False,crop_size=args.image_size, few_shot=True, seed=repeat, delete_empty_masks=delete_empty_masks, target_cls=args.cls, if_spatial=args.if_spatial)

        trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=num_workers)
        valloader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=num_workers)

        final_score = train_model(trainloader,valloader,testloader,args.dir_checkpoint,args.epochs)
        final_score_list.append(final_score)

        print(final_score_list)
        print('****')

    print(final_score_list)
