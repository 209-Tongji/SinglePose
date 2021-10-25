
import torch
import torchvision
import numpy as np

from HRNet import HRNet
from ResPose import PoseResNet
from RegressFlow import RegressFlow
from mscoco import MSCOCO
from aistpose2d import AISTPose2D

from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import transforms
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt

from config import update_config, opt

cfg = update_config(opt.cfg)

def load_pretrained(model, pretrained_path, device):
    if device == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    else:
        pretrained_dict = torch.load(pretrained_path, map_location='cuda:0')
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if v.shape == pretrained_dict[k].shape:
            model_dict[k] = pretrained_dict[k]
        else:
            print('size mismatch in %s. Expect %s but get %s' %(k, v.shape, pretrained_dict[k].shape))
    model.load_state_dict(model_dict)
    print("Successfully load pre-trained model from %s " %pretrained_path)
    return model

def person_detection(img_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()

    #resize = transforms.Resize((720, 1080))
    to_tensor = transforms.ToTensor()
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #torch内置FasterRCNN会对图像进行标准化

    #origin_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    origin_img = Image.open(img_path, mode='r').convert('RGB')
    #img = resize(origin_img)
    img = to_tensor(origin_img)
    img = img.to(device)
    
    output = model(img.unsqueeze(0))
    #print(output)

    labels = output[0]['labels'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    bboxes = output[0]['boxes'].cpu().detach().numpy()
    person_index = list(set(np.argwhere(labels==1).squeeze(axis=1).tolist()) & set(np.argwhere(scores>0.9).squeeze(axis=1).tolist()))
    
    #print(person_index)
    '''
    annotated_image = resize(origin_img)
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype('../Arial.ttf', 15)

    for i in person_index:
        box_location = bboxes[i].tolist()
        draw.rectangle(xy=box_location, outline='blue')
        draw.rectangle(xy=[l + 1. for l in box_location], outline='blue')

        text_size = font.getsize('Bachelor'.upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill='blue')
        draw.text(xy=text_location, text='Bachelor', fill='white', font=font)
        print(box_location)

    del draw

    annotated_image.save('../temp/res.png')
    '''
    
    return origin_img, bboxes[person_index]

def change_box_scale(bbox, aspect_ratio=0.75):
    xmin, ymin , xmax, ymax= bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.zeros((2), dtype=np.float32)
    center[0] = xmin + w * 0.5
    center[1] = ymin + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    bbox[0] = center[0] - w * 0.5
    bbox[1] = center[1] - h * 0.5
    bbox[2] = center[0] + w * 0.5
    bbox[3] = center[1] + h * 0.5

    return bbox

def make_input(img_path, bboxes, idx=0):
    bbox = bboxes[idx].tolist()
    
    bbox = change_box_scale(bbox)
    origin_img = Image.open(img_path, mode='r').convert('RGB')
    img = TF.crop(origin_img, bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0])
    img.save("bbox0.png")
    resize = transforms.Resize((256, 192))
    to_tensor = transforms.ToTensor()
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406])
    #img = normalize(to_tensor(resize(img)))
    img = resize(img)
    img.save("bbox1.png")
    img = to_tensor(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)
    #img = normalize(img)
    '''
    xmin, ymin, xmax, ymax = bbox
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, 0.75)
    scale = scale * 1.0

    input_size = (256, 192)
    inp_h, inp_w = input_size

    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) 
    img = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    bbox = _center_scale_to_box(center, scale)

    cv2.imwrite("bbox.png", img)

    img = im_to_torch(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)
    '''
    return img, bbox

    return img, bbox

def recover_coords(pred_jts):
    coords = pred_jts.cpu().detach().numpy()
    coords = coords.astype(float)
    #pred_scores = pred_scores.cpu().numpy()
    #pred_scores = pred_scores.astype(float)

    coords[:, 0] = (coords[:, 0] + 0.5) * (192 // 48)
    coords[:, 1] = (coords[:, 1] + 0.5) * (256 // 64)

    return coords

def heatmap_to_coord(batch_heatmaps):
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps.device)

    preds[:, :, 0] = idx % width  # column
    preds[:, :, 1] = torch.floor(idx / width)  # row

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals

def output_to_coords(output, idx=0):
    pred_jts = output.pred_jts[idx]
    coords = pred_jts.cpu().detach().numpy()
    coords = coords.astype(float)
    coords[:, 0] = (coords[:, 0] + 0.5) * 192
    coords[:, 1] = (coords[:, 1] + 0.5) * 256
    return coords

def draw_joints(origin_img, coords, bbox):
    #print(origin_img.size)
    #origin_img.save("origin.png")
    img = TF.crop(origin_img, bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0])
    img = TF.resize(img, (256,192))
    #img.save("bbox.png")
    print(coords)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(img)
    ax.scatter(coords[:,0], coords[:,1])
    fig.savefig("./res.png")
    plt.close()


def inference(img_path):
    if cfg.MODEL.TYPE == 'HRNet':
        model = HRNet(32, 17, 0.1)
    elif cfg.MODEL.TYPE == 'RegressFlow':
        model = RegressFlow(cfg=cfg)
    model = load_pretrained(model, cfg.INFERENCE.PRETRAINED, device)
    model.to(device)
    model.eval()

    origin_img, bboxes = person_detection(img_path)
    img, bbox = make_input(img_path, bboxes, idx=0)
    print(bbox) #[280.48983001708984, 139.78399658203125, 490.63077545166016, 419.971923828125]
    img = img.to(device)
    #print(img)
    output = model(img.unsqueeze(0))

    if cfg.MODEL.TYPE == 'RegressFlow':
        coords = output_to_coords(output)
    else:
        preds, maxvals = heatmap_to_coord(output)
        coords = recover_coords(preds[0])
    #coords = process_output(output, bbox)
    draw_joints(origin_img, coords, bbox)

if __name__ == '__main__':
    inference("./temp/0521.png")
