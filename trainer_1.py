if __name__ == '__main__':
    import copy
    import torch
    from src.models.modnet import MODNet
    from src.trainer import soc_adaptation_iter
    from torch.utils.data import DataLoader
    from setting import BS, LR, EPOCHS, SEMANTIC_SCALE, DETAIL_SCALE, MATTE_SCALE, SAVE_EPOCH_STEP
    from matting_dataset import MattingDataset, Rescale, \
            ToTensor, Normalize, ToTrainArray, \
            ConvertImageDtype, GenTrimap
    from torchvision import transforms



    modnet = torch.nn.DataParallel(MODNet())
    ckp_pth = './pretrained/modnet_photographic_portrait_matting.ckpt'
    if torch.cuda.is_available():
            modnet = modnet.cuda()
            weights = torch.load(ckp_pth)
    else:
            weights = torch.load(ckp_pth, map_location=torch.device('gpu'))
    modnet.load_state_dict(weights)  # NOTE: please finish this function

    transform = transforms.Compose([
        Rescale(512),
        GenTrimap(),
        ToTensor(),
        ConvertImageDtype(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTrainArray()
    ])
    mattingDataset = MattingDataset(transform=transform)

    optimizer = torch.optim.Adam(modnet.parameters(), lr=LR, betas=(0.9, 0.99))
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * EPOCHS), gamma=0.1)
    dataloader = DataLoader(mattingDataset,
                            batch_size=BS,
                            shuffle=False)  # NOTE: please finish this function

    for epoch in range(0, EPOCHS):
        print(f'epoch: {epoch}/{EPOCHS - 1}')
        backup_modnet = copy.deepcopy(modnet)
        for idx, (image) in enumerate(dataloader):
            soc_semantic_loss, soc_detail_loss = soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
            print(f'{(idx + 1) * BS}/{len(mattingDataset)} --- '
                  f'soc_semantic_loss: {soc_semantic_loss:f}, soc_detail_loss: {soc_detail_loss:f}\r',
                  end='')
        # lr_scheduler.step()
    torch.save(modnet.state_dict(), f'pretrained/modnet_custom_portrait_matting_last_epoch_weight.ckpt')