from torchvision.datasets import ImageFolder
from torchvision import transforms
from config import setting
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as Data
import torch
import time
import os


def data_process(cfg):

    data_test = ImageFolder(root=cfg.test_root,transform=transform)

   
    x_test= Data.DataLoader(dataset=data_test,batch_size=cfg.batch_size,num_workers=cfg.num_workers)
    return x_test


def model_test_all(model,x):

    device = cfg.device

    num = 0

    each_test_acc_total = 0


    model = model.to(device)

    model.eval()

    print("开始推理验证")

    start_time = time.time()


    with torch.no_grad():


        for step, (b_x,b_y) in enumerate(x):

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
                
            pre_lab = torch.argmax(output,dim=1)  
        
            num += b_x.size(0) 
                
            each_test_acc_total += torch.sum(pre_lab == b_y)  


    acc = (each_test_acc_total/num).double().item()

    end_time = time.time()

    time_total = end_time - start_time

    minutes, seconds = divmod(time_total, 60)

    print(f"推理共耗时: {int(minutes)} 分 {seconds:.1f} 秒")

    return acc








def load_pic(path,cfg):
    try:
        img = Image.open(path).convert('RGB')  # 确保 RGB 三通道
    except Exception as e:
        print(f"无法读取图片: {e}")
        return None
    

    img = transform(img).unsqueeze(0)  # 增加 batch 维度，形状: (1, 3, 224, 224) 

    return img

def model_test_single(cfg):

    device = cfg.device

    print(device)

    model = cfg.model.to(device)

    model.eval()

    results = []

    image_extensions = ('.jpg', '.jpeg', '.png','webp')
    for fname in os.listdir(cfg.single_root):
        if fname.lower().endswith(image_extensions):
            img_path = os.path.join(cfg.single_root, fname)
            img_tensor = load_pic(img_path, cfg)
            if img_tensor is None:
                continue
            x = img_tensor.to(device)

            with torch.no_grad():
                output = model(x)
                prob = F.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(prob, 1)
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
                results.append({
                'filename': fname,
                'predicted': classes[predicted_idx],         #可显示指定cfg.class_names
                'confidence': confidence
            })

    return results













if __name__ == '__main__':

    cfg = setting()

    transform =  transforms.Compose([
        transforms.Resize((cfg.input_size,cfg.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 均值
                             std=[0.229, 0.224, 0.225])])

    checkpoint = torch.load(cfg.save_path,map_location=cfg.device)

    cfg.model.load_state_dict(checkpoint['model_state_dict'])

    classes = ImageFolder(root=cfg.data_root).classes

    if cfg.predict_mode == 0:

        x = data_process(cfg)

        acc = model_test_all(cfg.model,x)

        print(f"准确率是{acc:.4f}")

    elif cfg.predict_mode == 1:

        results = model_test_single(cfg)

        for res in results:
            print(f"{res['filename']:20} -> {res['predicted']} (置信度: {res['confidence']:.4f})")

