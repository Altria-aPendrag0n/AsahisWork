from maxvit import *
from maxvit import MaxViT
from tkinter import * 
from tkinter import ttk
import tkinter.messagebox
import tkinter as tk
from PIL import Image, ImageTk 
import time
import cv2
import numpy as np
from matplotlib.figure import Figure
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os

class my_MaxViT(nn.Module):

    def __init__(self):
        super(my_MaxViT, self).__init__()
        model = MaxViT(
               depths=(2, 2, 5, 2), channels=(96, 128, 256, 512), embed_dim=64,
            num_classes=73,
        )
        self.model = model

    def forward(self, x):
        return self.model(x)


net = my_MaxViT()

class_label = ['ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL', 'AFRICAN CROWNED CRANE', 'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH', 'AFRICAN OYSTER CATCHER', 'AFRICAN PIED HORNBILL', 'AFRICAN PYGMY GOOSE', 'ALBATROSS', 'ALBERTS TOWHEE', 'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH', 'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN DIPPER', 'AMERICAN FLAMINGO', 'AMERICAN GOLDFINCH', 'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART', 'AMERICAN ROBIN', 'AMERICAN WIGEON', 'AMETHYST WOODSTAR', 'ANDEAN GOOSE', 'ANDEAN LAPWING', 'ANDEAN SISKIN', 'ANHINGA', 'ANIANIAU', 'ANNAS HUMMINGBIRD', 'ANTBIRD', 'ANTILLEAN EUPHONIA', 'APAPANE', 'APOSTLEBIRD', 'ARARIPE MANAKIN', 'ASHY STORM PETREL', 'ASHY THRUSHBIRD', 'ASIAN CRESTED IBIS', 'ASIAN DOLLARD BIRD', 'ASIAN GREEN BEE EATER', 'ASIAN OPENBILL STORK', 'AUCKLAND SHAQ', 'AUSTRAL CANASTERO', 'AUSTRALASIAN FIGBIRD', 'AVADAVAT', 'AZARAS SPINETAIL', 'AZURE BREASTED PITTA', 'AZURE JAY', 'AZURE TANAGER', 'AZURE TIT', 'BAIKAL TEAL', 'BALD EAGLE', 'BALD IBIS', 'BALI STARLING', 'BALTIMORE ORIOLE', 'BANANAQUIT', 'BAND TAILED GUAN', 'BANDED BROADBILL', 'BANDED PITA', 'BANDED STILT', 'BAR-TAILED GODWIT', 'BARN OWL', 'BARN SWALLOW', 'BARRED PUFFBIRD', 'BARROWS GOLDENEYE', 'BAY-BREASTED WARBLER', 'BEARDED BARBET', 'BEARDED BELLBIRD', 'BEARDED REEDLING', 'BELTED KINGFISHER', 'BIRD OF PARADISE']
label_dict = {
    'ABBOTTS BABBLER': '阿氏噪鹛，东南亚热带森林的小型鸟类，绿色羽毛，群居生活，主食昆虫，因栖息地破坏数量减少，濒危。',
    'ABBOTTS BOOBY': '阿氏鹈鹕，大型海鸟，常见于热带海域，善潜水捕食鱼类，长嘴大喉囊是其特征，受人类活动影响，数量下降。',
    'ABYSSINIAN GROUND HORNBILL': '阿比西尼亚地犀鸟，大型鸟类，分布于非洲草原，以昆虫和小型动物为食，长喙用于挖掘和捕食，是当地生态的重要组成部分。',
    'AFRICAN CROWNED CRANE': '非洲冕鹤，优雅的大型涉禽，身披灰白羽毛，头顶红冠，常见于非洲的湿地和草原，是非洲大陆的标志性物种之一。',
    'AFRICAN EMERALD CUCKOO': '非洲翠鴗，色彩艳丽的森林鸟类，以水果和昆虫为食，其亮丽的羽毛和悦耳的鸣声使其成为观鸟者的热门观察对象。',
    'AFRICAN FIREFINCH': '非洲火雀，鲜艳的小型鸟类，常见于非洲的开阔地和灌丛，雄鸟羽毛鲜红夺目，是鸟类爱好者喜爱的观赏鸟种。',
    'AFRICAN OYSTER CATCHER': '非洲蛎鹬，长嘴长腿的涉禽，分布于非洲的海岸线和淡水湖泊，以贝类和其他水生动物为食，是湿地生态系统的重要指示物种。',
    'AFRICAN PIED HORNBILL': '非洲斑犀鸟，大型森林鸟类，羽毛黑白相间，长喙用于敲击果实，主要分布于非洲的热带雨林中，是森林生态的关键物种。',
    'AFRICAN PYGMY GOOSE': '非洲侏儒雁，小型水禽，分布于非洲的淡水湖泊和沼泽地，体型虽小但善于游泳和潜水，是当地水生生态的重要成员。',
    'ALBATROSS': '信天翁，大型海洋鸟类，拥有强健的翅膀和卓越的飞行能力，常在海上长时间滑翔，以鱼类、乌贼和甲壳类为食。',
    'ALBERTS TOWHEE': '艾伯特鸫，小型鸟类，分布于北美洲的森林和灌丛地带，以昆虫和种子为食，其悦耳的鸣声是森林中的一道美丽风景。',
    'ALEXANDRINE PARAKEET': '亚历山大鹦鹉，色彩鲜艳的中型鹦鹉，主要分布于亚洲的热带和亚热带地区，以其善于模仿人类语言和清脆的叫声而著称。',
    'ALPINE CHOUGH': '高山红尾鸲，生活在高山地区的鸟类，羽毛以红色和黑色为主，善于在岩石间寻找食物，是高山生态系统的重要指示物种。',
    'ALTAMIRA YELLOWTHROAT': '阿尔塔米拉黄喉鹀，小型鸣禽，分布于中美洲的热带雨林中，以其明亮的黄色喉部和悦耳的鸣声而知名，常在林间穿梭觅食昆虫。',
    'AMERICAN AVOCET': '美洲勺嘴鹬，涉禽，长有独特的上翘嘴型，以小型水生动物为食，常见于美洲的湿地和沿海地区，是湿地生态系统的重要保护对象。',
    'AMERICAN BITTERN': '美洲麻鳽，一种大型涉禽，常见于美洲的淡水湿地。羽毛灰褐色，长嘴适于捕食鱼类和两栖动物，是湿地生态系统的重要成员。',
    'AMERICAN COOT': '美洲秧鸡，水禽，分布广泛。身体圆润，善于游泳和潜水，以水生植物和小型动物为食，是淡水生态系统的重要组成部分。',
    'AMERICAN DIPPER': '美洲水鹨，小型鸣禽，常在溪流边活动。善于在水中觅食昆虫，其强健的腿部使其能在湍急的溪流中稳定站立。',
    'AMERICAN FLAMINGO': '美洲火烈鸟，大型涉禽，以其鲜艳的粉红色羽毛而著名。主要分布于美洲的热带和亚热带地区，以水生动物和小型昆虫为食。',
    'AMERICAN GOLDFINCH': '美洲金翅雀，小型鸣禽，以其亮丽的金黄色羽毛和悦耳的鸣声而知名。以种子和昆虫为食，常见于美洲的开阔地带和灌丛。',
    'AMERICAN KESTREL': '美洲红尾鸲，小型猛禽，以其敏捷的飞行和捕食技巧而著称。羽毛以蓝色和棕色为主，长尾巴有助于飞行中的平衡和机动性。',
    'AMERICAN PIPIT': '美洲鹨，小型鸣禽，广泛分布于美洲的草原和开阔地。羽毛灰褐色，善于在地面上奔跑和觅食昆虫。',
    'AMERICAN REDSTART': '美洲红尾鸲，小型鸟类，以其亮丽的红色和黑色羽毛而知名。常在林间穿梭觅食昆虫，是森林生态系统的重要成员。',
    'AMERICAN ROBIN': '美洲知更鸟，中型鸣禽，以其鲜艳的胸部羽毛和悦耳的鸣声而受到人们喜爱。常在地面觅食昆虫和果实，是城市和乡村的常见鸟类。',
    'AMERICAN WIGEON': '美洲角鹬，小型水禽，以其独特的喙形和优雅的泳姿而著名。常在浅水区域觅食水生植物和小型动物。',
    'AMETHYST WOODSTAR': '紫晶林星，小型蜂鸟，以其紫罗兰色的羽毛和长喙而知名。主要分布于中、南美洲的热带雨林中，以花蜜和小型昆虫为食。',
    'ANDEAN GOOSE': '安第斯雁，大型水禽，分布于南美洲的高山湖泊和河流。羽毛灰白相间，善于游泳和飞行，是高山生态系统的重要成员。',
    'ANDEAN LAPWING': '安第斯鹬，中型涉禽，以其独特的黑白相间羽毛和醒目的红色腿而著称。常在开阔地和草原活动，以昆虫和小型动物为食。',
    'ANDEAN SISKIN': '安第斯金翅雀，小型鸣禽，以其亮丽的金黄色羽毛和悦耳的鸣声而受到人们喜爱。主要分布于南美洲的高山地区，以种子和昆虫为食。',
    'ANHINGA': '蛇鹈，大型水鸟，以其独特的捕鱼方式而知名。常在水中潜伏，突然伸出长嘴捕捉鱼类，羽毛防水性好，便于在水中活动。',
    'ANIANIAU': '（注：此鸟名可能拼写错误或为非标准名称，无法提供准确信息）',
    'ANNAS HUMMINGBIRD': '安娜蜂鸟，小型蜂鸟，以其小巧的身形和快速的飞行而著称。羽毛绚丽多彩，以花蜜和小型昆虫为食，常见于北美洲的热带和亚热带地区。',
    'ANTILLEAN EUPHONIA': '安地列斯噪鹛，中型鸣禽，以其悦耳的鸣声和亮丽的羽毛而受到人们喜爱。主要分布于加勒比海地区的岛屿，以果实和种子为食。',
    'APAPANE': '阿帕帕内鸟，小型鸣禽，以其鲜艳的红色羽毛和悦耳的鸣声而知名。常见于夏威夷群岛的森林和灌丛地带，以花蜜和昆虫为食。',
    'APOSTLEBIRD': '（注：此鸟名可能拼写错误或为非标准名称，无法提供准确信息）',
    'ARARIPE MANAKIN': '阿拉里佩鸲，小型鸟类，以其独特的求偶行为和鲜艳的羽毛而受到关注。主要分布于巴西的阿拉里佩地区，是当地特有的物种。',
    'ASHY STORM PETREL': '灰暴风鹱，一种小型海鸟，以其灰褐色的羽毛和强健的飞行能力著称。常在海上飞行，以鱼类和浮游生物为食，是海洋生态系统的重要成员。',
    'ASHY THRUSHBIRD': '灰噪鹛，一种常见于澳大利亚的小型鸟类。羽毛以灰色为主，常在灌丛和森林中活动，以昆虫和果实为食，是当地生态系统中不可或缺的一环。',
    'ASIAN CRESTED IBIS': '亚洲冠麻鳽，一种大型涉禽，以其独特的冠羽和优雅的姿态而知名。主要分布于亚洲的湿地和沼泽地区，以鱼类、两栖动物和昆虫为食。',
    'ASIAN DOLLARD BIRD': '（注：此鸟名可能拼写错误或为非标准名称，无法提供准确信息）',
    'ASIAN GREEN BEE EATER': '亚洲绿蜂虎，以其鲜艳的羽毛和捕食蜂类的习性而著称。常见于亚洲的热带和亚热带地区，常在树木上筑巢，以蜂类和其他昆虫为食。',
    'ASIAN OPENBILL STORK': '亚洲开嘴鹳，一种大型涉禽，以其独特的开嘴觅食方式和优雅的体态而知名。主要分布于亚洲的湿地和沼泽地带，以鱼类、蛙类和软体动物为食。',
    'AUCKLAND SHAQ': '（注：此鸟名可能拼写错误或为非标准名称，无法提供准确信息）',
    'AUSTRAL CANASTERO': '澳洲鹪鹩，一种小型鸣禽，以其悦耳的鸣声和活泼的性格而受到人们喜爱。主要分布于澳大利亚的森林和灌丛地带，以昆虫和种子为食。',
    'AUSTRALASIAN FIGBIRD': '澳亚无花果鸟，一种中型鸟类，以其对无花果的偏好而知名。羽毛色彩鲜艳，常在无花果树上觅食和筑巢，是澳大利亚特有的物种之一。',
    'AVADAVAT': '红腹灰雀，以其醒目的红色腹部和灰褐色的背部而著名。分布于南亚和东南亚地区，常在灌丛和开阔地带活动，以种子和昆虫为食。',
    'AZARAS SPINETAIL': '阿扎拉棘尾雀，一种小型鸟类，以其独特的棘状尾羽和活泼的性格而知名。主要分布于南美洲的森林和灌丛地带，以昆虫和果实为食。',
    'AZURE BREASTED PITTA': '蓝胸鹟，一种色彩鲜艳的鸟类，以其蓝色的胸部和亮丽的羽毛而著称。分布于东南亚的热带雨林中，常在林间穿梭觅食昆虫。',
    'AZURE JAY': '蓝松鸦，一种大型鸣禽，以其鲜艳的蓝色羽毛和悦耳的鸣声而知名。常成群活动于森林和山地，以果实、种子和昆虫为食。',
    'AZURE TANAGER': '蓝鹟，一种色彩亮丽的鸟类，以其蓝色的羽毛和悦耳的鸣叫声而受到人们喜爱。分布于中美洲的热带雨林中，以果实和昆虫为食。',
    'AZURE TIT': '蓝山雀，一种小型鸣禽，以其蓝色的羽毛和活泼的性格而著称。常在树林和灌丛中活动，以昆虫和种子为食，是森林生态系统的重要成员。',
    'BAIKAL TEAL': '贝加尔湖鹬，一种中型涉禽，以其独特的繁殖习性和对环境的适应性而知名。主要分布于俄罗斯贝加尔湖地区，以水生动物和昆虫为食。',
    'BALD EAGLE': '白头海雕，北美洲的国鸟，以其强壮的体态和敏锐的视力而著称。羽毛以白色和棕色为主，头部有白色羽毛，常在湖泊和河流上空盘旋，以鱼类和其他水生动物为食。',
    'BALD IBIS': '秃鹳，一种大型涉禽，以其裸露的头部和长颈而知名。主要分布于非洲和亚洲的干旱地区，以昆虫、小型哺乳动物和鸟类的尸体为食。',
    'BALI STARLING': '巴厘岛星椋鸟，一种小型鸟类，以其独特的羽毛图案和悦耳的鸣声而受到人们喜爱。主要分布于印度尼西亚的巴厘岛，常在森林和灌丛中活动，以果实和种子为食。',
    'BALTIMORE ORIOLE': '巴尔的摩黄鹂，一种中型鸣禽，以其亮丽的黄色羽毛和悦耳的鸣声而知名。分布于北美洲的东部和中部地区，常在树木上筑巢，以昆虫和果实为食。',
    'BAY-BREASTED WARBLER': '湾胸鹟莺，是一种小巧而活泼的鸟类，以其醒目的胸部斑纹和悦耳的鸣声而知名。常见于北美洲的森林和灌丛地带，以昆虫和种子为食。',
    'BEARDED BARBET': '须鹩哥，一种体型中等的鸟类，以其独特的须状羽毛和强健的喙部而著称。常见于亚洲和非洲的热带雨林中，以果实和昆虫为食。',
    'BEARDED BELLBIRD': '须钟鸟，一种以其独特钟状鸣声和华丽羽毛而闻名的鸟类。分布于中美洲和南美洲的热带森林中，常在树上活动，以果实和昆虫为食。',
    'BEARDED REEDLING': '须苇莺，一种小型鸣禽，以其纤细的体态和优雅的飞行姿态而知名。常栖息于淡水湿地和沼泽地带，以昆虫和小型水生动物为食。',
    'BELTED KINGFISHER': '带冕翠鸟，一种色彩鲜艳的鸟类，以其独特的蓝色羽毛和冕状头饰而著称。常见于北美洲的河流、湖泊等水域附近，以鱼类为食，善于潜水捕食。',
    'BIRD OF PARADISE': '天堂鸟，以其绚丽多彩的羽毛和独特的求偶舞蹈而闻名于世。分布于新几内亚等地的热带雨林中，是鸟类中的艺术大师。',
    'BANANAQUIT': '香蕉雀，一种色彩鲜艳的小型鸟类，以其对香蕉花的偏好而知名。常见于中美洲和南美洲的热带花园和果园中，以花蜜和昆虫为食。',
    'BAND TAILED GUAN': '环尾雉，一种体型较大的鸟类，以其长长的尾羽和华丽的羽毛而著称。分布于南美洲的热带雨林中，常在地面觅食，以植物果实和种子为食。',
    'BANDED BROADBILL': '纹阔嘴鸟，以其独特的阔嘴和身上醒目的条纹而知名。分布于亚洲的热带雨林中，以果实和昆虫为食，常在林间穿梭。',
    'BANDED PITA': '纹翠鸟，一种色彩艳丽的鸟类，以其身上的条纹和鲜艳的羽毛而著称。常见于东南亚的热带雨林中，以昆虫和小型动物为食。',
    'BANDED STILT': '带鹬，一种涉禽，以其修长的腿和独特的带状羽毛而知名。常栖息于湖泊、河流等淡水湿地中，以小鱼、虾和昆虫为食。',
    'BAR-TAILED GODWIT': '斑尾鹬，一种中等体型的涉禽，以其独特的斑纹尾羽和优雅的飞行姿态而著称。广泛分布于北半球的湿地和沿海地区，以软体动物和昆虫为食。',
    'BARN OWL': '仓小鸮，一种夜行性鸟类，以其独特的面部特征和静谧的飞行方式而知名。常在农田、草原和森林地带活动，以小型哺乳动物和昆虫为食。',
    'BARN SWALLOW': '家燕，一种广泛分布的候鸟，以其优雅的飞行姿态和长长的尾羽而著称。常在人类居住区附近筑巢，以昆虫为食，是夏季的常见鸟类。',
    'BARRED PUFFBIRD': '斑鸫鸠，一种中等体型的鸟类，以其身上的斑纹和独特的鸣声而知名。分布于中南美洲的热带雨林中，常在树上活动，以果实和昆虫为食。',
    'BARROWS GOLDENEYE': '巴氏金眼鹬，一种迷人的小型水鸟，以其醒目的金色眼圈和优雅的姿态而著称。常见于淡水湖泊和河流中，以水生昆虫和小鱼为食。',
    'BALTIMORE ORIOLE': '巴尔的摩黄鹂，以其亮丽的橙色羽毛和优美的歌声而知名。主要分布于北美洲的东部地区，常见于公园和森林中，以昆虫和果实为食。',
    'BANANAQUIT': '香蕉雀，一种活泼的小型鸟类，以其对香蕉的喜爱而得名。羽毛色彩鲜艳，常在热带地区的花园和果园中活动，以花蜜和昆虫为食。',
    'BALD EAGLE': '白头海雕，北美洲的国鸟，以其强健的体态和雄壮的飞行姿态而著称。羽毛主要为白色和棕色，头部和尾部羽毛为白色，以其雄伟的外貌和强大的力量而广受人们喜爱。',
    'BALD IBIS': '秃鹳，一种大型鸟类，以其独特的秃头和粉色的皮肤而知名。主要分布于非洲和亚洲的干旱地区，以昆虫、小型哺乳动物和鸟蛋为食，常在开阔地带活动。',
}


net = my_MaxViT()
net.load_state_dict(torch.load("maxvit-net.pt"))
def predict(img):
    channel_mean = torch.Tensor([0.485,0.456,0.406])
    channel_std = torch.Tensor([0.229,0.224,0.225])
    transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=channel_mean, std=channel_std)
]) 
    img1 = transform_fn(img)
    input = img1.unsqueeze(0)
    with torch.no_grad():
        output = net(input)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class_label = class_label[predicted_class_index]
    print(f"预测类别：{predicted_class_label}")
    print(f"概率：{probabilities[predicted_class_index].item()}")
    return predicted_class_label,probabilities[predicted_class_index].item()

top = Tk()
top.title("鸟类识别系统")
top.geometry("1000x700")
top.configure(bg = "#F8CBAD")  # 窗口背景颜色
    # 大标题
Label(top ,text = "鸟类识别系统",font = ("宋体",18),fg = "black",bg = "#F8CBAD",width = 35,height = 2).place(x = 300,y = 0)

def show_image(frame ,img_path = './bk1.jpg'):
    global photo1  
    photo_open1 = Image.open(img_path) 
    print(photo_open1)
    photo1 = photo_open1.resize((224 ,224))  
    photo1 = ImageTk.PhotoImage(photo1)   
    Label(frame,image=photo1).place(x= 140,y = 60) 
    
def get_label():
    path = path_entry.get()
    print(path)
    if os.path.exists(path):
      tk.messagebox.showinfo('success', '上传成功!')
      img = Image.open(path)
      show_image(Frame2_1,img_path = path)
      label ,pro = predict(img)
      if pro<0.75:
        tk.messagebox.showinfo('注意', '该结果是正确的概率不足0.75')
      res = label_dict[label]
      text_box.delete('1.0',tk.END)
      text_box.insert('1.0',res)
    else: 
      tk.messagebox.showinfo('error', '上传失败!')
      
Frame2 = Frame(top, bg="#B4C7E7", height=625, width=700)
Frame2.place(x=160, y=50)  # 将 Frame2 的位置调整到距离左侧 200px，距离顶部 50px 处

Label(Frame2, text="图片路径", bg="#C1FFC1", font=("黑体", 15), width=12, height=1).place(x=150, y=50)
#Button(Frame2, text="退出系统", bg="#C1FFC1", font=("黑体", 15), width=12, height=2, command=top.destroy).place(x=350, y=20)
img_path = StringVar

path_entry = Entry(Frame2,textvariable=img_path,font = ('黑体',15),width = 20,)
path_entry.place(x = 280,y = 50)
Frame2_1 = Frame(Frame2, bg="white", height=500, width=600)
Frame2_1.place(x=50, y=100)
res = " "
text_content = (res)
x = '百灵鸟'
text_box = tk.Text(Frame2_1, font=("微软雅黑", 12), bg="white", width=50, height=5)
text_box.insert("1.0", text_content)
text_box.grid(row=0, column=0, padx=40, pady=300)
Label(Frame2_1, text="你上传的图片", font=("微软雅黑", 15), bg="white", width=18, height=2).place(x=150, y=10)
#tkinter.Text(Frame2_1, text=f"预测结果：阿氏噪鹛，又称阿氏鸟，是一种小型鸟类，主要分布于东南亚的热带和亚热带森林中。身披绿色或橄榄色羽毛，常成群活动，以昆虫为食。由于栖息地破坏及人类活动，其数量逐渐减少，属濒危物种。", bg="white", width = 100, height=2).place(x=40, y=300)
"""Label(Frame2_1, text="2号摊位需求量： ", font=("微软雅黑", 12), bg="#FFC727", width=50, height=2).place(x=20, y=250)
Label(Frame2_1, text="3号摊位需求量： ", font=("微软雅黑", 12), bg="#20EA37", width=15, height=2).place(x=20, y=350)"""
Button(Frame2, text="确认", bg="white", font=("黑体", 11), width=4, height=1, command=get_label).place(x=490, y=50)
 
top.mainloop()  #消息循环
