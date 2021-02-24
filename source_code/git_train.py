from git_init import *
from git_config import *


network_id = networkName + networkType + identifier

TBlogger    = pl_loggers.TensorBoardLogger('logs/' + network_id)

trainer = pl.Trainer(logger=TBlogger, log_every_n_steps=5, gpus=1, max_epochs=nrEpochs)

if __name__ == '__main__':
    trainer.fit(net, train_dataloader=train_data_loader, val_dataloaders=val_data_loader)
    trainer.test(net, test_dataloaders=test_data_loader)
