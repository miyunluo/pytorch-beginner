from torch.autograd import Variable
from torchvision import transforms
from load_img import load_img, show_img
from run_code import run_style_transfer

style_img = load_img('./picture/style0.png')
style_img = Variable(style_img)
content_img = load_img('./picture/content1.jpg')
content_img = Variable(content_img)

input_img = content_img.clone()
out = run_style_transfer(content_img, style_img, input_img, num_epoches=80)

show_img(out.cpu())

save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))

save_pic.save('./picture/saved_picture.png')
