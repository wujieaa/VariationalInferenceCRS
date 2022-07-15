import matplotlib.pyplot as plt
import pickle
import os.path as path
import argparse
import os

FINAL_OUTPUTS_ITEM = 'item'
FINAL_OUTPUTS_EXPLAINATION = 'explaination'
FINAL_OUTPUTS_CRITIQUING = 'critiquing'
FINAL_OUTPUTS_MINLOSS = 'minloss'
FINAL_OUTPUTS_AVGLOSS = 'avgloss'
FINAL_OUTPUTS_RATING_AVGLOSS = 'rating_avgloss'
FINAL_OUTPUTS_KEYPHRASE_AVGLOSS = 'keyphrase_avgloss'

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

def main(args):
    file_name='bvae4_result_m6'
    result_path = path.abspath(path.join(args.result_dir,file_name+'.pkl'))
    if path.exists(result_path):
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print('no data is found')
        return
    base_dir = r'd:\data'
    if not path.exists(path.join(base_dir, file_name)):
        os.mkdir(path.join(base_dir, file_name))
    item_result=data[FINAL_OUTPUTS_ITEM]
    ndcgs=list(map(lambda x: x['NDCG'][0], item_result))
    plot1(ndcgs,title='ndcg',legend=['ndcg'],save_path=path.join(base_dir,file_name,'ndcg.png'))
    R_Precision = list(map(lambda x: x['R-Precision'][0], item_result))
    plot1(R_Precision, title='R-Precision', legend=['R-Precision'], save_path=path.join(base_dir,file_name,'R-Precision.png'))
    clicks = list(map(lambda x: x['Clicks'][0], item_result))
    plot1(clicks, title='Clicks', legend=['Clicks'], save_path=path.join(base_dir,file_name,'Clicks.png'))
    tmpK = 5
    atk = []
    while tmpK < args.topk:
        atk.append(tmpK)
        tmpK += 5
    atk.append(args.topk)
    for k in atk:
        recall = list(map(lambda x: x[f'Recall@{k}'][0], item_result))
        plot1(recall, title=f'Recall@{k}', legend=[f'Recall@{k}'], save_path=path.join(base_dir,file_name,f'Recall@{k}.png'))
        precision = list(map(lambda x: x[f'Precision@{k}'][0], item_result))
        plot1(precision, title=f'Precision@{k}', legend=[f'Precision@{k}'], save_path=path.join(base_dir,file_name, f'Precision@{k}.png'))
        map_arr = list(map(lambda x: x[f'MAP@{k}'][0], item_result))
        plot1(map_arr, title=f'MAP@{k}', legend=[f'MAP@{k}'], save_path=path.join(base_dir,file_name, f'MAP@{k}.png'))
    #-----------explanation------------
    explaination_result = data[FINAL_OUTPUTS_EXPLAINATION]
    recall = list(map(lambda x: x[f'Recall@{args.topk_keyphrase}'][0], explaination_result))
    plot1(recall, title=f'Recall@{args.topk_keyphrase}', legend=[f'Recall@{args.topk_keyphrase}'], save_path=path.join(base_dir, file_name,f'explanation-Recall@{args.topk_keyphrase}.png'))
    precision = list(map(lambda x: x[f'Precision@{args.topk_keyphrase}'][0], explaination_result))
    plot1(precision, title=f'Precision@{args.topk_keyphrase}', legend=[f'Precision@{args.topk_keyphrase}'],save_path=path.join(base_dir, file_name,f'explanation-Precision@{args.topk_keyphrase}.png'))
    #-----------critiquing------------
    critiquing_result = data[FINAL_OUTPUTS_CRITIQUING]
    keyphrase_topk_array = [5, 10, 20]
    for k in keyphrase_topk_array :
        f_map = list(map(lambda x: x[f'F-MAP@{k}'][0], critiquing_result))
        plot1(f_map, title=f'F-MAP@{k}', legend=[f'F-MAP@{k}'], save_path=path.join(base_dir,file_name,f'F-MAP@{k}.png'))
    #----------loss-------------------
    avg_loss_result = data[FINAL_OUTPUTS_AVGLOSS]
    plot1(avg_loss_result, title='avg_loss', legend=['avg_loss'],
          save_path=path.join(base_dir, file_name,'avg_loss.png'))
    min_loss_result = data[FINAL_OUTPUTS_MINLOSS]
    plot1(min_loss_result, title='min_loss', legend=['min_loss'],
          save_path=path.join(base_dir,file_name, 'min_loss.png'))
    rating_loss_result = data[FINAL_OUTPUTS_RATING_AVGLOSS]
    plot1(rating_loss_result, title='rating_avg_loss', legend=['rating_loss'],
          save_path=path.join(base_dir, file_name,'rating_avg_loss.png'))
    keyphrase_loss_result = data[FINAL_OUTPUTS_KEYPHRASE_AVGLOSS]
    plot1(keyphrase_loss_result, title='keyhprase_loss', legend=['keyhprase_loss'],
          save_path=path.join(base_dir, file_name,'keyhprase_loss.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='../../../../lightning_logs')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--topk_keyphrase', type=int, default=10)
    args = parser.parse_args()
    print('------args-----------')
    for k in list(vars(args).keys()):
        print(f'{k}:{vars(args)[k]}')
    print('------args-----------')
    main(args)