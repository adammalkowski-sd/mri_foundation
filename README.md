# A New Foundation Model for MRI

This is the official code for our paper: [???](https://arxiv.org/abs/2404.09957), where we propose a new foundation model designed specifically for MRI. Figure 1 shows the overview of this model, including training data, training algorithm, and performance on the few-shot segmentation task. 

![Fig1: Overview of general fine-tuning strategies based on different levels of dataset availability.](https://github.com/mazurowski-lab/finetune-SAM/blob/main/finetune_strategy_v9.png)

The model's weights can be accessed [here](https://drive.google.com/file/d/1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW/view?usp=sharing).

To load the segmentation model with pre-trained weights, check this code snippet:
```
from models.sam import sam_model_registry  
import cfg

args = cfg.parse_args()
model = sam_model_registry['vit_b'](args, checkpint="PATH_TO_CHECKPOINT", num_classes=args.num_cls, image_size=args.image_size, pretrained_sam=True)

# Forward
img_emb = model.image_encoder(imgs)
sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=None, masks=None)
pred, _ = model.mask_decoder(image_embeddings=img_emb,                                                              
                             image_pe=sam.prompt_encoder.get_dense_pe(),                                            
                             sparse_prompt_embeddings=sparse_emb,                                                   
                             dense_prompt_embeddings=dense_emb,                                                     
                             multimask_output=True)
```
