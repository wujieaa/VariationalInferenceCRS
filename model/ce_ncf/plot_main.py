import matplotlib.pyplot as plt
import pickle
import os.path as path
from model.ce_ncf.ce_ncf_lightning import CENCFLightning
from model.ce_ncf.params import *

def plot(data1, data2, *, position=111, title='title', legend, save_path):
    plt.figure(1)
    plt.subplot(position)
    plt.plot(list(range(len(data1))), data1, 'r-',
             list(range(len(data2))), data2, 'b-.')
    plt.title(title)
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.savefig(save_path + '-' + title + '.png')
    plt.close()
def plot1(data, *, position=111, title='title', legend, save_path):
    plt.figure(1)
    plt.subplot(position)
    plt.plot(list(range(len(data))), data, 'r-')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.legend(legend)
    plt.savefig(save_path)
    plt.close()

def main():
    result_path = path.abspath('../../../../lightning_logs/cencf_result.pkl')
    if (path.exists(result_path)):
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print('no data is found')
        return
    item_result=data[CENCFLightning.FINAL_OUTPUTS_ITEM]
    base_dir=r'd:\data'
    ndcgs=list(map(lambda x: x['NDCG'][0], item_result))
    plot1(ndcgs,title='ndcg',legend=['ndcg'],save_path=path.join(base_dir,'ndcg.png'))
    R_Precision = list(map(lambda x: x['R-Precision'][0], item_result))
    plot1(R_Precision, title='R-Precision', legend=['R-Precision'], save_path=path.join(base_dir,'R-Precision.png'))
    clicks = list(map(lambda x: x['Clicks'][0], item_result))
    plot1(clicks, title='Clicks', legend=['Clicks'], save_path=path.join(base_dir,'Clicks.png'))
    tmpK = 5
    atk = []
    while tmpK < topk:
        atk.append(tmpK)
        tmpK += 5
    atk.append(topk)
    for k in atk:
        recall = list(map(lambda x: x[f'Recall@{k}'][0], item_result))
        plot1(recall, title=f'Recall@{k}', legend=[f'Recall@{k}'], save_path=path.join(base_dir,f'Recall@{k}.png'))
        precision = list(map(lambda x: x[f'Precision@{k}'][0], item_result))
        plot1(precision, title=f'Precision@{k}', legend=[f'Precision@{k}'], save_path=path.join(base_dir, f'Precision@{k}.png'))
        map_arr = list(map(lambda x: x[f'MAP@{k}'][0], item_result))
        plot1(map_arr, title=f'MAP@{k}', legend=[f'MAP@{k}'], save_path=path.join(base_dir, f'MAP@{k}.png'))
    #-----------explanation------------
    explaination_result = data[CENCFLightning.FINAL_OUTPUTS_EXPLAINATION]
    recall = list(map(lambda x: x[f'Recall@{topk_keyphrase}'][0], explaination_result))
    plot1(recall, title=f'Recall@{topk_keyphrase}', legend=[f'Recall@{topk_keyphrase}'], save_path=path.join(base_dir, f'explanation-Recall@{topk_keyphrase}.png'))
    precision = list(map(lambda x: x[f'Precision@{topk_keyphrase}'][0], explaination_result))
    plot1(precision, title=f'Precision@{topk_keyphrase}', legend=[f'Precision@{topk_keyphrase}'],save_path=path.join(base_dir, f'explanation-Precision@{topk_keyphrase}.png'))
    #-----------critiquing------------
    critiquing_result = data[CENCFLightning.FINAL_OUTPUTS_CRITIQUING]
    keyphrase_topk_array = [5, 10, 20]
    for k in keyphrase_topk_array :
        f_map = list(map(lambda x: x[f'F-MAP@{k}'][0], critiquing_result))
        plot1(f_map, title=f'F-MAP@{k}', legend=[f'F-MAP@{k}'], save_path=path.join(base_dir,f'F-MAP@{k}.png'))
    #----------loss-------------------
    avg_loss_result = data[CENCFLightning.FINAL_OUTPUTS_AVGLOSS]
    plot1(avg_loss_result, title='avg_loss', legend=['avg_loss'],
          save_path=path.join(base_dir, 'avg_loss.png'))
    min_loss_result = data[CENCFLightning.FINAL_OUTPUTS_MINLOSS]
    plot1(min_loss_result, title='min_loss', legend=['min_loss'],
          save_path=path.join(base_dir, 'min_loss.png'))
    rating_loss_result = data[CENCFLightning.FINAL_OUTPUTS_RATING_AVGLOSS]
    plot1(rating_loss_result, title='rating_avg_loss', legend=['rating_loss'],
          save_path=path.join(base_dir, 'rating_avg_loss.png'))
    keyphrase_loss_result = data[CENCFLightning.FINAL_OUTPUTS_KEYPHRASE_AVGLOSS]
    plot1(keyphrase_loss_result, title='keyhprase_loss', legend=['keyhprase_loss'],
          save_path=path.join(base_dir, 'keyhprase_loss.png'))
    recon_loss_result = data[CENCFLightning.FINAL_OUTPUTS_RECON_AVGLOSS]
    plot1(recon_loss_result, title='recon_loss', legend=['recon_loss'],
          save_path=path.join(base_dir, 'recon_loss.png'))


if __name__ == '__main__':
    main()