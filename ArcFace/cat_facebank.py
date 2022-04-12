import numpy as np
import torch


# 加载pth,npy文件中存储的特征
def load_facebank(facebank_path):
    embeddings = torch.load(facebank_path + '/facebank.pth',
                            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    names = np.load(facebank_path + '/names.npy')
    embeddings2 = torch.load(facebank_path + '/facebank_1.pth',
                             map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    names2 = np.load(facebank_path + '/names_1.npy')
    p_arr = np.concatenate((names, ['wtx_3']))  # 先将p_变成list形式进行拼接，注意输入为一个tuple
    return torch.cat((embeddings, embeddings2), 0), p_arr


if __name__ == '__main__':
    a = load_facebank('./facebank')
    print(a[0].shape, a[0], a[1])
    torch.save(a[0], './facebank' + '/facebank.pth')
    np.save('./facebank/names', a[1])