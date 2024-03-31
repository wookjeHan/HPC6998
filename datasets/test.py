from spider import Spider
from dialogsum import DialogSum
from e2e_nlg import E2ENLG
from utils import create_dataloader

spider = Spider(download=True)
print(spider.__getitem__(1))
print(spider.__len__())
dialogsum = DialogSum(download=True)
print(dialogsum.__getitem__(1))
print(dialogsum.__len__())
e2e_nlg = E2ENLG(download=True)
print(e2e_nlg.__getitem__(1))
print(e2e_nlg.__len__())

spider_dataloader = create_dataloader(
    dataset=Spider,
    root=f"./{Spider.name}",
    train=False,
    batch_size=32,
    shuffle=True,
    num_workers=8
)
