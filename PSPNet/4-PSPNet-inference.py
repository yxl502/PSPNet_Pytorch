from utils.dataloader import make_datapath_list, DataTransform

rootpath = './data/VOCdevkit/VOC2012/'

train_img_list, train_anno_list, val_img_list, val_anno_list = \
    make_datapath_list(rootpath)

from utils.pspnet import PSPNet

net = PSPNet(n_classes=21)

import torch
state_dict = torch.load('./weights/pspnet50_30.pth',
                        map_location=torch.device('cpu'))

net.load_state_dict(state_dict)
print('网络设置完毕：成功载入了训练完毕的权重。')


# image_file_path = './data/cowboy-757575_640.jpg'



# 需要转RGB模式```


from PIL import Image
# img = Image.open(image_file_path)


image_file_path = './data/d2.jpg'
# image_file_path = './data/x2x.png'
img = Image.open(image_file_path).convert('RGB')

img_width, img_height = img.size

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()


color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

anno_file_path = val_anno_list[0]

anno_class_img = Image.open(anno_file_path)

p_palette = anno_class_img.getpalette()

phase = 'val'

img, anno_class_img = transform(phase, img, anno_class_img)

net.eval()

x = img.unsqueeze(0)

outputs = net(x)

y = outputs[0]

y = y[0].detach().numpy()

import numpy as np
y = np.argmax(y, axis=0)

anno_class_img = Image.fromarray(np.uint8(y), mode='P')

anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)

anno_class_img.putpalette(p_palette)

plt.imshow(anno_class_img)
plt.show()


trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
anno_class_img = anno_class_img.convert('RGBA')

for x in range(img_width):
    for y in range(img_height):
        pixel = anno_class_img.getpixel((x, y))
        r, g, b, a = pixel

        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            continue

        else:
            trans_img.putpixel((x, y), (r, g, b ,150))


img = Image.open(image_file_path)

result = Image.alpha_composite(img.convert('RGBA'), trans_img)

plt.imshow(result)
plt.show()