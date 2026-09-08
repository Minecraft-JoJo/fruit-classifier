from model import ResNet
import torch
class setting:
    def __init__(self):
        self.model = ResNet(num_classes=50,
                    in_channels=3
                    )

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print('硬件:',self.device)
              

    '''
    num_classes = 10       #分类数
    in_channels = 1        #输入通道数

    '''  


    # --- 数据参数 ---
    data_root = './BIG/fruit-classifier/data/train'       # 数据存放根目录
    val_root = './BIG/fruit-classifier/data/val'
    test_root = './BIG/fruit-classifier/data/test'
    single_root = './BIG/fruit-classifier/data/single'

    save_path = './BIG/fruit-classifier/save/checkpoint_all.pth'
    interrupted_save_path1 = './BIG/fruit-classifier/save/interrupted_best_model.pth'
    interrupted_save_path2 = './BIG/fruit-classifier/save/checkpoint_half.pth'
    log_path = './train_log/train.csv'


    batch_size = 100
    num_workers = 10            # 数据加载的并行进程数
    split = 0.8                #训练和验证集中,训练集的占有量
    input_size=224           #默认图片大小      googlenet 224,VGG 227,
    class_names = ["Apple","Avocado","Banana","Beetroot","Blackberry","Blueberry","Broccoli","Cabbage","Capsicum","Carrot","Cauliflower","Chilli Peper","Corn","Cucumber","Dates","Dragonfruit","Eggplant","Fig","Garlic","Ginger","Grapes","Guava","Jalepeno","Kiwi","Lemon","Lettuce","Mango","Mushroom","Okra","Olive","Onion","Orange","Paprika","Peas","Pear","Peanuts","Pineapple","Pomegranate","Potato","Pumpkin","Raddish","Rambutan","Soy Beans","Spinach","Strawberry","Sweetcorn","Sweetpotato","Tomato","Turnip","Watermelon"]

    # --- 训练参数 ---
    epochs = 5               # 训练轮数，也设置了续训练的轮次
    learning_rate = 0.0001
    weight_decay = 5e-4        # L2 正则化系数

    predict_mode = 1          #预测模式。0是测试集，1是单个测试



    train_mode = 1  
    '''
    是否开启续训练模式。
    0:不开启续训练模式，第一次训练
    1开启续训练,并加载0模式的完整训练模型
    2开启续训练,并加载上次中断训练的模型

    '''



    

  