import torch
import os
import torch.nn.functional as F
import yaml
import argparse, pickle, random
import os.path as osp

from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms, datasets
from diffusers import UNet2DModel
from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from glob import glob
from diffusers import DDPMPipeline
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset


class Classification_2d(Dataset):

    def __init__(self, dataset_name, transform, mode):
        self.image_dir = osp.join('outfitdata_set3_4598', mode)
        self.transform = transform
        self.mode = mode

        if dataset_name == 'fashion':
            self.outfit_list = glob(osp.join(self.image_dir, '*'))
            self.data_list = list()

            for outfit_id_path in self.outfit_list:
                for i in range(1, 6):
                    data_path = osp.join(outfit_id_path, f'{i}.jpg')
                    self.data_list.append(data_path)
        elif dataset_name == "character_a":
            self.data_list = glob('./character_edge2color/train_A/*.png')
        elif dataset_name == "character_b":
            self.data_list = glob('./character_edge2color/train_B/*.png')

    def __getitem__(self, index):
        img_path = self.data_list[index]
        target_image = Image.open(osp.join(img_path))
        target_image = target_image.convert('RGB')
        return {"images": self.transform(target_image)}

    def __len__(self):
        return len(self.data_list)

def get_loader(config):

    transform_train = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = Classification_2d(config['dataset_name'], transform_train, 'train')

    train_loader = DataLoader(trainset,
                              batch_size=config['train_batch_size'],
                              num_workers=2,
                              pin_memory=True)

    return train_loader

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config['eval_batch_size'],
        generator=torch.Generator(device='cpu').manual_seed(config['seed']), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config['output_dir'], "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        #log_with="tensorboard",
        project_dir=os.path.join(config['output_dir'], "logs"),
    )
    if accelerator.is_main_process:
        if config['output_dir'] is not None:
            os.makedirs(config['output_dir'], exist_ok=True)
        if config['push_to_hub']:
            repo_id = create_repo(
                repo_id=config['hub_model_id'] or Path(config['output_dir']).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config['save_image_epochs'] == 0 or epoch == config['num_epochs'] - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config['save_model_epochs'] == 0 or epoch == config['num_epochs'] - 1:
                if config['push_to_hub']:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config['output_dir'],
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config['output_dir'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='specifies config yaml file')

    params = parser.parse_args()

    assert os.path.exists(params.config)
    config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
    dataset_name = config['dataset_name']

    if dataset_name == "huggan/smithsonian_butterflies_subset":
        dataset = load_dataset(config['dataset_name'], split="train")
        dataset.set_transform(transform)
        preprocess = transforms.Compose(
            [
                transforms.Resize((config['image_size'], config['image_size'])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True)
    else:
        train_dataloader = get_loader(config)


    model = UNet2DModel(
        sample_size=config['image_size'],  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config['lr_warmup_steps'],
        num_training_steps=(len(train_dataloader) * config['num_epochs']),
    )

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
