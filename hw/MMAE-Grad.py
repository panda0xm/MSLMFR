import torch
import numpy as np
import scipy.io
import torchvision
import matplotlib.pyplot as plt # plotting library
import pandas as pd 
from torch import nn
from evaluate import evaluate
import dataset
size1=32
size2=100
size3=100
size4=100
#2.0版使用了新的损失函数
class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.sharelinear1=torch.nn.Linear(size1*5+6, size2)
        self.sharelinear2=torch.nn.Linear(size2, size3)
        self.linear00=torch.nn.Linear(216, size1)
        self.linear10=torch.nn.Linear(76, size1)
        self.linear20=torch.nn.Linear(64, size1)
        self.linear30=torch.nn.Linear(6, 6)
        self.linear40=torch.nn.Linear(240, size1)
        self.linear50=torch.nn.Linear(47, size1)
                
    def forward(self, x0,x1,x2,x3,x4,x5):
        x0 = (self.linear00(x0))
        x1 = (self.linear10(x1))
        x2 = (self.linear20(x2))
        x3 = (self.linear30(x3))
        x4 = (self.linear40(x4))
        x5 = (self.linear50(x5))
        x_hidden=torch.cat((x0,x1,x2,x3,x4,x5),1)
        x_hidden=(self.sharelinear1(x_hidden))
        #x_hidden=torch.relu(self.sharelinear2(x_hidden))
        return x_hidden

class Decoder(nn.Module):
    
    def __init__(self,size3,size4,size_out):
        super().__init__()
        self.linear01=torch.nn.Linear(size3, size4)
        self.output02=torch.nn.Linear(size4, size_out)
         
    def forward(self, x):
        #x = (self.linear01(x))
        x = self.output02(x)        
        return x


### Training function
def train_epoch(device,batch_size,epoch):
    ### Define the loss function
    loss_fn0 = torch.nn.MSELoss()
    loss_fn1 = torch.nn.MSELoss()
    loss_fn2 = torch.nn.MSELoss()
    loss_fn3 = torch.nn.MSELoss()
    loss_fn4 = torch.nn.MSELoss()
    loss_fn5 = torch.nn.MSELoss()
    
    import Aligned_MTL
    mtl=Aligned_MTL.Aligned_MTL()
    mtl.encoder=(encoder)
    mtl.task_num=6
    mtl.device=device
    mtl.rep_grad=False

    import GradNorm
    grad=GradNorm.GradNorm()   
    grad.encoder=(encoder)        
    grad.task_num=6
    grad.device=device
    grad.init_param()
    grad.rep_grad=False
    grad.epoch=epoch
        
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder0.train()
    decoder1.train()
    decoder2.train()
    decoder3.train()
    decoder4.train()
    decoder5.train()
    train_loss0 = []
    train_loss1 = []
    train_loss2 = []
    train_loss3 = []
    train_loss4 = []
    train_loss5 = []
    #data0_iter.shuffle    
    data0_iter,data1_iter,data2_iter,data3_iter,data4_iter,data5_iter,labels_iter=dataset.Handwritten_numerals(True)
    #因为不足一个batch没做处理，所以最好可以整除
    batch_num=len(data0_iter)/batch_size    
    
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for i in range(int(batch_num)): 
        image_batch0=torch.tensor(np.array(data0_iter[i*batch_size:(i+1)*batch_size]))
        image_batch1=torch.tensor(np.array(data1_iter[i*batch_size:(i+1)*batch_size]))
        image_batch2=torch.tensor(np.array(data2_iter[i*batch_size:(i+1)*batch_size]))
        image_batch3=torch.tensor(np.array(data3_iter[i*batch_size:(i+1)*batch_size]))
        image_batch4=torch.tensor(np.array(data4_iter[i*batch_size:(i+1)*batch_size]))
        image_batch5=torch.tensor(np.array(data5_iter[i*batch_size:(i+1)*batch_size]))
        #image_batch6=data6_iter[i*batch_size,(i+1)*batch_size]
        # Move tensor to the proper device
        image_batch0 = image_batch0.to(device)
        image_batch1 = image_batch1.to(device)
        image_batch2 = image_batch2.to(device)
        image_batch3 = image_batch3.to(device)
        image_batch4 = image_batch4.to(device)
        image_batch5 = image_batch5.to(device)
        
        
        '''
        params_to_optimize = [
            {'params': encoder.parameters()},
            {'params': decoder0.parameters()},
            {'params': decoder1.parameters()},
            {'params': decoder2.parameters()},
            {'params': decoder3.parameters()},
            {'params': decoder4.parameters()},
            {'params': decoder5.parameters()}
        ]
        optimizer = torch.optim.Adam(params_to_optimize, lr=0.001, weight_decay=1e-05)
        
        #task0
        optimizer.zero_grad()        
        encoded_data = encoder(image_batch0,image_batch1,image_batch2,image_batch3,image_batch4,image_batch5)
        decoded_data0 = decoder0(encoded_data)
        loss0 = loss_fn0(decoded_data0, image_batch0)
        loss0.backward()
        optimizer.step()
        
        #task1
        optimizer.zero_grad()
        encoded_data = encoder(image_batch0,image_batch1,image_batch2,image_batch3,image_batch4,image_batch5)
        decoded_data1 = decoder1(encoded_data)
        loss1 = loss_fn1(decoded_data1, image_batch1)
        loss1.backward()
        optimizer.step()
        
        #task2
        optimizer.zero_grad()
        encoded_data = encoder(image_batch0,image_batch1,image_batch2,image_batch3,image_batch4,image_batch5)
        decoded_data2 = decoder2(encoded_data)
        loss2 = loss_fn2(decoded_data2, image_batch2)
        loss2.backward()
        optimizer.step()
        
        #task3
        optimizer.zero_grad()
        encoded_data = encoder(image_batch0,image_batch1,image_batch2,image_batch3,image_batch4,image_batch5)
        decoded_data3 = decoder3(encoded_data)
        loss3 = loss_fn3(decoded_data3, image_batch3)
        loss3.backward()
        optimizer.step()
        
        #task4
        optimizer.zero_grad()
        encoded_data = encoder(image_batch0,image_batch1,image_batch2,image_batch3,image_batch4,image_batch5)
        decoded_data4 = decoder4(encoded_data)
        loss4 = loss_fn4(decoded_data4, image_batch4)
        loss4.backward()
        optimizer.step()
        
        #task5
        optimizer.zero_grad()
        encoded_data = encoder(image_batch0,image_batch1,image_batch2,image_batch3,image_batch4,image_batch5)
        decoded_data5 = decoder5(encoded_data)
        loss5 = loss_fn5(decoded_data5, image_batch5)
        loss5.backward()
        optimizer.step()
        '''
        params_to_optimize1 = [
            {'params': encoder.parameters()},
            {'params': decoder0.parameters()},
            {'params': decoder1.parameters()},
            {'params': decoder2.parameters()},
            {'params': decoder3.parameters()},
            {'params': decoder4.parameters()},
            {'params': decoder5.parameters()}
        ]
        optimizer1 = torch.optim.Adam(params_to_optimize1, lr=0.001, weight_decay=1e-05)
        optimizer1.zero_grad()
        encoded_data = encoder(image_batch0,image_batch1,image_batch2,image_batch3,image_batch4,image_batch5)
        decoded_data0 = decoder0(encoded_data)
        loss0 = loss_fn0(decoded_data0, image_batch0)
        decoded_data1 = decoder1(encoded_data)
        loss1 = loss_fn1(decoded_data1, image_batch1)
        decoded_data2 = decoder2(encoded_data)
        loss2 = loss_fn2(decoded_data2, image_batch2)
        decoded_data3 = decoder3(encoded_data)
        loss3 = loss_fn3(decoded_data3, image_batch3)
        decoded_data4 = decoder4(encoded_data)
        loss4 = loss_fn4(decoded_data4, image_batch4)
        decoded_data5 = decoder5(encoded_data)
        loss5 = loss_fn5(decoded_data5, image_batch5)
        train_loss_buffer[:, epoch] = [loss0.detach().cpu().numpy(),
            loss1.detach().cpu().numpy(),
            loss2.detach().cpu().numpy(),
            loss3.detach().cpu().numpy(),
            loss4.detach().cpu().numpy(),
            loss5.detach().cpu().numpy()]    
        grad.train_loss_buffer=train_loss_buffer
        alpha=grad.backward(losses=[loss0,loss1,loss2,loss3,loss4,loss5],alpha=1.5)
        #print('alpha',alpha)
        optimizer1.step()
        
        
        '''
        import basic_balancer
        balancer = basic_balancer.BasicBalancer()
        balancer.step(losses=[loss0,loss1,loss2,loss3,loss4,loss5],
                  shared_params=encoder.parameters(),
                  task_specific_params={decoder0.parameters(),decoder1.parameters(),decoder2.parameters(),decoder3.parameters(),decoder4.parameters(),decoder5.parameters()},
                  shared_representation=encoded_data,
                  last_shared_layer_params=None)
        optimizer.step()
        '''

        
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss0.append(loss0.detach().cpu().numpy())
        train_loss1.append(loss1.detach().cpu().numpy())
        train_loss2.append(loss2.detach().cpu().numpy())
        train_loss3.append(loss3.detach().cpu().numpy())
        train_loss4.append(loss4.detach().cpu().numpy())
        train_loss5.append(loss5.detach().cpu().numpy())
    return [np.mean(train_loss0),np.mean(train_loss1),np.mean(train_loss2),
    np.mean(train_loss3),np.mean(train_loss4),np.mean(train_loss5)]


def test_epoch(n_clusters):
    encoder.eval()
    encoded_samples = []
    data0_iter,data1_iter,data2_iter,data3_iter,data4_iter,data5_iter,labels_iter=dataset.Handwritten_numerals(False)
    for i in range(len(data0_iter)):
        data0 = torch.tensor(data0_iter[i]).unsqueeze(0).to(device)
        data1 = torch.tensor(data1_iter[i]).unsqueeze(0).to(device)
        data2 = torch.tensor(data2_iter[i]).unsqueeze(0).to(device)
        data3 = torch.tensor(data3_iter[i]).unsqueeze(0).to(device)
        data4 = torch.tensor(data4_iter[i]).unsqueeze(0).to(device)
        data5 = torch.tensor(data5_iter[i]).unsqueeze(0).to(device)
        label = labels_iter[i]
        # Encode image
        with torch.no_grad():
            encoded_img  = encoder(data0,data1,data2,data3,data4,data5)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    encoded_samples

    X=encoded_samples.drop(['label'],axis=1)
    Y=encoded_samples.label.astype(int)
    #print(X)
    #降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)
    #展示真实标签
    # 绘制散点图
    plt.figure('truth') 
    plt.subplot(221)
    scatter=plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=Y, alpha=0.7)
    plt.xlabel('tsne-2d-one')
    plt.ylabel('tsne-2d-two')
    plt.title('TSNE Results')
    plt.legend(*scatter.legend_elements())
     
    #聚类，#展示预测标签
    from sklearn.cluster import KMeans
    y_pred = KMeans(n_clusters=n_clusters,n_init=10).fit_predict(X)
    acc,nmi,purity,f1,precision,recall,ari=evaluate(Y,y_pred)
    print(f'\nkmeans:acc:{acc},nmi:{nmi},purity:{purity},f1:{f1},precision:{precision},recall:{recall},ari:{ari}')
    #plt.figure('kmeans')    
    plt.subplot(222)
    plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=y_pred, alpha=0.7)
    #plt.legend(*scatter.legend_elements())
    
    from sklearn.mixture import GaussianMixture
    # 创建GMM对象并拟合数据#展示预测标签    
    gmm = GaussianMixture(n_components=n_clusters, max_iter=100)
    gmm.fit(X)
    y_pred3 = gmm.predict(X)   
    acc,nmi,purity,f1,precision,recall,ari=evaluate(Y,y_pred3)    
    print(f'\ngmm:acc:{acc},nmi:{nmi},purity:{purity},f1:{f1},precision:{precision},recall:{recall},ari:{ari}')
    #plt.figure('gmm')
    plt.subplot(224)
    plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=y_pred3, alpha=0.7)
    #plt.legend(*scatter.legend_elements())
    #plt.show()
   
    from sklearn.cluster import SpectralClustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    spectral.fit(X)
    y_pred2 = spectral.labels_
    acc,nmi,purity,f1,precision,recall,ari=evaluate(Y,y_pred2)
    print(f'\nspectral:acc:{acc},nmi:{nmi},purity:{purity},f1:{f1},precision:{precision},recall:{recall},ari:{ari}')
    #plt.figure('spectral')
    plt.subplot(223)
    plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1], c=y_pred2, alpha=0.7)
    #plt.legend(*scatter.legend_elements())
    plt.show()
    return acc,nmi,purity,f1,precision,recall,ari
    
import os  # 导入os模块     
def train(num_epochs,batch_size):
    diz_loss = {'loss0':[],'loss1':[],'loss2':[],'loss3':[],'loss4':[],'loss5':[]}
    max_nmi=0.8
    for epoch in range(1,num_epochs+1):
        [loss0,loss1,loss2,loss3,loss4,loss5] =train_epoch(device, batch_size,epoch)
        if(epoch%100==0):
            acc,nmi,purity,f1,precision,recall,ari=test_epoch(10)
            if(nmi>max_nmi):
                max_nmi=nmi
                # 保存模型
                if not os.path.exists('model'):
                    os.mkdir('model')
                torch.save(encoder.state_dict(), 'model/encoder.pth')
                torch.save(decoder0.state_dict(), 'model/decoder0.pth')
                torch.save(decoder1.state_dict(), 'model/decoder1.pth')
                torch.save(decoder2.state_dict(), 'model/decoder2.pth')
                torch.save(decoder3.state_dict(), 'model/decoder3.pth')
                torch.save(decoder4.state_dict(), 'model/decoder4.pth')
                torch.save(decoder5.state_dict(), 'model/decoder5.pth')
                os.rename('model', f'acc{acc},nmi{nmi},purity{purity}')
                plt.show()
            print(f'\n EPOCH {epoch}/{num_epochs} \t loss0 {loss0}\t loss1 {loss1}\t loss2 {loss2}\t loss3 {loss3}\t loss4 {loss4}\t loss5 {loss5}')
        diz_loss['loss0'].append(loss0)
        diz_loss['loss1'].append(loss1)
        diz_loss['loss2'].append(loss2)
        diz_loss['loss3'].append(loss3)
        diz_loss['loss4'].append(loss4)
        diz_loss['loss5'].append(loss5)

    # Plot losses
    plt.figure('loss')
    #plt.figure(figsize=(10,8))
    plt.semilogy(diz_loss['loss0'], label='loss0')
    plt.semilogy(diz_loss['loss1'], label='loss1')
    plt.semilogy(diz_loss['loss2'], label='loss2')
    plt.semilogy(diz_loss['loss3'], label='loss3')
    plt.semilogy(diz_loss['loss4'], label='loss4')
    plt.semilogy(diz_loss['loss5'], label='loss5')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid()
    plt.legend()
    plt.title('loss')
    plt.show()

encoder = Encoder()
decoder0 = Decoder(size3,size4,216)
decoder1 = Decoder(size3,size4,76)  
decoder2 = Decoder(size3,size4,64)  
decoder3 = Decoder(size3,size4,6)  
decoder4 = Decoder(size3,size4,240)  
decoder5 = Decoder(size3,size4,47)  
num_epochs = 500
batch_size=400
train_loss_buffer = np.zeros([6, num_epochs+1])

if __name__ == "__main__":    
    data0_iter,data1_iter,data2_iter,data3_iter,data4_iter,data5_iter,labels_iter=dataset.Handwritten_numerals(False)
    #raise Exception()
    # Check if the GPU is available
    # Move both the encoder and the decoder to the selected device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    encoder.to(device)
    decoder0.to(device)
    decoder1.to(device)
    decoder2.to(device)
    decoder3.to(device)
    decoder4.to(device)
    decoder5.to(device)
    
    train(num_epochs,batch_size)
    
    '''    
    # 保存模型
    torch.save(encoder.state_dict(), 'model/encoder.pth')
    torch.save(decoder0.state_dict(), 'model/decoder0.pth')
    torch.save(decoder1.state_dict(), 'model/decoder1.pth')
    torch.save(decoder2.state_dict(), 'model/decoder2.pth')
    torch.save(decoder3.state_dict(), 'model/decoder3.pth')
    torch.save(decoder4.state_dict(), 'model/decoder4.pth')
    torch.save(decoder5.state_dict(), 'model/decoder5.pth')
    #load
    encoder.load_state_dict(torch.load('model acc0.921,nmi0.8605069018686414,purity0.921/encoder.pth'))
    decoder0.load_state_dict(torch.load('model acc0.921,nmi0.8605069018686414,purity0.921/decoder0.pth'))
    decoder1.load_state_dict(torch.load('model acc0.921,nmi0.8605069018686414,purity0.921/decoder1.pth'))
    decoder2.load_state_dict(torch.load('model acc0.921,nmi0.8605069018686414,purity0.921/decoder2.pth'))
    decoder3.load_state_dict(torch.load('model acc0.921,nmi0.8605069018686414,purity0.921/decoder3.pth'))
    decoder4.load_state_dict(torch.load('model acc0.921,nmi0.8605069018686414,purity0.921/decoder4.pth'))
    decoder5.load_state_dict(torch.load('model acc0.921,nmi0.8605069018686414,purity0.921/decoder5.pth'))
    encoder.to(device)
    decoder0.to(device)
    decoder1.to(device)
    decoder2.to(device)
    decoder3.to(device)
    decoder4.to(device)
    decoder5.to(device)'''
    test_epoch(10)